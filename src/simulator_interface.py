"""
Simulator interface for the causal-splice perturbation instrument: builds the
PushCenterMultiLite RoboSuite scene, provides camera-geometry helpers
(extrinsics-based lateral/depth directions, world->pixel projection), exact
simulator state snapshot/restore, and the render -> preprocess -> tokenize
path that mirrors the original iVideoGPT training/inference preprocessing
(third_party/iVideoGPT: ivideogpt/data/simple_dataloader.py and
inference/utils.py -- images/255 then torchvision.transforms.functional.resize
to 64x64, not torch.nn.functional.interpolate).
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF

from .push_center_multi_lite import PushCenterMultiLite
from robosuite.utils import camera_utils

CAMERA_NAME = "agentview_shift_2"
RENDER_RES = 256  # matches renderer_camera_height/width in vp2/scripts/configs/env/robosuite.yaml
TARGET_RES = 64   # iVideoGPT checkpoint resolution

CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "OSC_POSITION",
            "input_max": 1, "input_min": -1,
            "output_max": [0.05, 0.05, 0.05], "output_min": [-0.05, -0.05, -0.05],
            "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    },
}


def build_env(seed=0):
    env = PushCenterMultiLite(
        robots="Panda",
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERA_NAME,
        camera_heights=RENDER_RES,
        camera_widths=RENDER_RES,
        control_freq=1.0,
        seed=seed,
    )
    env.reset()
    return env


def object_joint_name(env, obj_index=0):
    """obj_index 0 = 'cube' (push_center_multi.py's primary/largest object,
    0.06 half-size box -- also object_index=0 / 'large_box' in the fork's
    per-object pushing reward variants). This is our calibration object.
    """
    return env.objects[obj_index].joints[0]


def get_qpos_addr(env, obj_index=0):
    """Returns (start, end) qpos indices for the object's free joint, via
    sim.model.get_joint_qpos_addr -- NOT guessed. Free joint qpos layout is
    [x, y, z, qw, qx, qy, qz] (7 values).
    """
    name = object_joint_name(env, obj_index)
    addr = env.sim.model.get_joint_qpos_addr(name)
    return addr, name


def snapshot_state(env):
    """Full state needed for deterministic restoration: qpos, qvel, time
    (via MjSim.get_state()) plus ctrl and act explicitly (not covered by
    MjSimState in this robosuite version). We never call mj_step() in this
    calibration, so ctrl/act do not affect mj_forward()'s outputs, but we
    snapshot them anyway per the calibration protocol.
    """
    sim_state = env.sim.get_state()
    return {
        "time": sim_state.time,
        "qpos": sim_state.qpos.copy(),
        "qvel": sim_state.qvel.copy(),
        "ctrl": env.sim.data.ctrl.copy(),
        "act": env.sim.data.act.copy() if hasattr(env.sim.data, "act") else None,
    }


def restore_state(env, snap):
    from robosuite.utils.binding_utils import MjSimState
    env.sim.set_state(MjSimState(time=snap["time"], qpos=snap["qpos"].copy(), qvel=snap["qvel"].copy()))
    env.sim.data.ctrl[:] = snap["ctrl"]
    if snap["act"] is not None and hasattr(env.sim.data, "act"):
        env.sim.data.act[:] = snap["act"]
    env.sim.forward()  # mujoco.mj_forward -- NEVER mj_step (see module docstring)


def get_object_world_pos(env, obj_index=0):
    return env.sim.data.body_xpos[env.object_body_ids[obj_index]].copy()


def set_object_world_pos(env, world_pos, obj_index=0):
    name = object_joint_name(env, obj_index)
    quat = env.sim.data.get_joint_qpos(name)[3:7].copy()
    env.sim.data.set_joint_qpos(name, np.concatenate([world_pos, quat]))
    env.sim.forward()


def camera_axes_world(env):
    """Camera-relative lateral-X, lateral/vertical-Y, and depth-Z axes,
    expressed in world coordinates, from the camera's actual extrinsic
    matrix (NOT assumed == world XYZ). OpenCV convention after robosuite's
    correction: column 0 = image +X (right), column 1 = image +Y (down),
    column 2 = camera forward / +depth (see camera_utils.get_camera_extrinsic_matrix).
    """
    R = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)
    return {"lateral_x": R[:3, 0].copy(), "lateral_y": R[:3, 1].copy(), "depth_z": R[:3, 2].copy()}


def world_to_pixel(env, world_point_xyz, render_res=RENDER_RES):
    """Projects a single 3D world point to pixel (row, col) in the
    render_res x render_res RENDERED (pre-resize) image, using the camera's
    actual intrinsic+extrinsic transform (robosuite.utils.camera_utils).
    """
    transform = camera_utils.get_camera_transform_matrix(env.sim, CAMERA_NAME, render_res, render_res)
    pix = camera_utils.project_points_from_world_to_camera(
        np.asarray(world_point_xyz).reshape(1, 3), transform, render_res, render_res
    )
    return pix[0]  # (row, col)


def render_raw(env):
    """Raw RENDER_RES x RENDER_RES uint8 RGB frame, OpenGL (bottom-up)
    convention as returned directly by sim.render() -- robosuite's default
    macros.IMAGE_CONVENTION is 'opengl' (no flip applied), verified by
    reading robosuite.utils.macros / robot_env.py:400. We apply the
    standard top-down display flip ourselves for viewing/orientation
    consistency; which convention the original iGibson-rendered training data
    used is not independently verified, part of the appearance-domain-shift
    caveat discussed in the paper's Limitations section.
    """
    img = env.sim.render(camera_name=CAMERA_NAME, width=RENDER_RES, height=RENDER_RES)
    return img[::-1].copy()


def preprocess_to_model_input(img_uint8_hwc):
    """Matches ivideogpt/data/simple_dataloader.py's data_augmentation() /
    inference/utils.py's NPZParser.preprocess(): images/255 then
    torchvision.transforms.functional.resize to TARGET_RES, operating on a
    (C,H,W) float tensor.
    """
    t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [TARGET_RES, TARGET_RES], antialias=True)
    return t  # (3, 64, 64) in [0,1]


@torch.no_grad()
def tokenize_single_frame_dyna(tokenizer, context_pixel_values, frame_pixel_chw, device):
    """Tokenizes one future frame against the given context frames (same
    isolated-frame tokenization trick used in src/causal_splice.py,
    justified there by compressive_vq_model.py's per-frame-independent dyna
    encoding). Returns the 16 dyna tokens (1,16) as int64.
    """
    clip = torch.cat([context_pixel_values, frame_pixel_chw.unsqueeze(0).unsqueeze(0).to(device)], dim=1)
    tokens, _ = tokenizer.tokenize(clip, tokenizer.context_length)
    prelude = tokenizer.context_length * (256 + 1) - 1
    return tokens[:, prelude + 1: prelude + 1 + 16].long()
