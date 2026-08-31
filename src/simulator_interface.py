"""
Simulator interface for the causal-splice perturbation instrument: env
construction, camera geometry, exact state snapshot/restore, and the
render/preprocess/tokenize path. Preprocessing matches iVideoGPT's own
training/inference pipeline (third_party/iVideoGPT:
ivideogpt/data/simple_dataloader.py, inference/utils.py): images/255 then
torchvision resize to 64x64, not torch.nn.functional.interpolate.
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
    """obj_index 0 is the cube (push_center_multi.py's primary object)."""
    return env.objects[obj_index].joints[0]


def get_qpos_addr(env, obj_index=0):
    """(start, end) qpos indices for the object's free joint: [x,y,z,qw,qx,qy,qz]."""
    name = object_joint_name(env, obj_index)
    addr = env.sim.model.get_joint_qpos_addr(name)
    return addr, name


def snapshot_state(env):
    """qpos/qvel/time plus ctrl/act, which MjSimState doesn't cover in this
    robosuite version. ctrl/act don't affect mj_forward's output since we
    never call mj_step, but are snapshotted for completeness.
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
    env.sim.forward()  # mj_forward, not mj_step


def get_object_world_pos(env, obj_index=0):
    return env.sim.data.body_xpos[env.object_body_ids[obj_index]].copy()


def set_object_world_pos(env, world_pos, obj_index=0):
    name = object_joint_name(env, obj_index)
    quat = env.sim.data.get_joint_qpos(name)[3:7].copy()
    env.sim.data.set_joint_qpos(name, np.concatenate([world_pos, quat]))
    env.sim.forward()


def camera_axes_world(env):
    """Lateral-X, lateral-Y, and depth-Z axes in world coordinates, from the
    camera's actual extrinsic matrix (not assumed equal to world XYZ).
    Columns are image +X, image +Y, camera-forward/+depth (OpenCV convention).
    """
    R = camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)
    return {"lateral_x": R[:3, 0].copy(), "lateral_y": R[:3, 1].copy(), "depth_z": R[:3, 2].copy()}


def world_to_pixel(env, world_point_xyz, render_res=RENDER_RES):
    """Projects a 3D world point to pixel (row, col) in the pre-resize,
    render_res x render_res rendered image.
    """
    transform = camera_utils.get_camera_transform_matrix(env.sim, CAMERA_NAME, render_res, render_res)
    pix = camera_utils.project_points_from_world_to_camera(
        np.asarray(world_point_xyz).reshape(1, 3), transform, render_res, render_res
    )
    return pix[0]  # (row, col)


def render_raw(env):
    """RENDER_RES x RENDER_RES uint8 RGB frame. sim.render() returns
    OpenGL (bottom-up) convention; we flip to top-down here.
    """
    img = env.sim.render(camera_name=CAMERA_NAME, width=RENDER_RES, height=RENDER_RES)
    return img[::-1].copy()


def preprocess_to_model_input(img_uint8_hwc):
    """Matches iVideoGPT's own preprocessing: images/255 then torchvision
    resize to TARGET_RES, as a (C,H,W) float tensor.
    """
    t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [TARGET_RES, TARGET_RES], antialias=True)
    return t  # (3, 64, 64) in [0,1]


@torch.no_grad()
def tokenize_single_frame_dyna(tokenizer, context_pixel_values, frame_pixel_chw, device):
    """Tokenizes one future frame against the given context (same
    isolated-frame trick as src/causal_splice.py). Returns the 16 dyna
    tokens as (1,16) int64.
    """
    clip = torch.cat([context_pixel_values, frame_pixel_chw.unsqueeze(0).unsqueeze(0).to(device)], dim=1)
    tokens, _ = tokenizer.tokenize(clip, tokenizer.context_length)
    prelude = tokenizer.context_length * (256 + 1) - 1
    return tokens[:, prelude + 1: prelude + 1 + 16].long()
