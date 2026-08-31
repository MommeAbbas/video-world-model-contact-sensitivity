"""
Offset-parameterized intervention for the event-centered temporal sensitivity
sweep (paper Section "Temporal Structure of Predictive Sensitivity"):
render_clean_pair_at and adaptive_epsilon_search_at are the same calibrated
instrument as matched_sensitivity.render_clean_pair / adaptive_epsilon_search,
except the intervention timestep t_i is passed in directly instead of being
computed as t_event-3, so the same instrument can be swept across
Delta_t in {-3,...,+3}. EPSILON_CANDIDATES_MM and TARGET_BAND are imported
unchanged, not redefined.
"""
import numpy as np

from . import simulator_interface as lib
from .matched_sensitivity import EPSILON_CANDIDATES_MM, TARGET_BAND, hamming_only
from robosuite.utils.binding_utils import MjSimState


def render_clean_pair_at(env, log, t_i, epsilon_mm, direction="lateral_x", obj_index=0):
    """obj_index: which env.objects[] entry to displace (default 0 = the cube
    studied in the paper). The displacement magnitude, direction convention,
    state-restore procedure, and render/tokenize path are identical regardless
    of obj_index."""
    qpos, qvel, tm = log["qpos"][t_i], log["qvel"][t_i], log["time"][t_i]

    env.sim.set_state(MjSimState(time=tm, qpos=qpos.copy(), qvel=qvel.copy()))
    env.sim.forward()
    obj_pos = lib.get_object_world_pos(env, obj_index)
    axes = lib.camera_axes_world(env)
    dir_vec = axes[direction] / np.linalg.norm(axes[direction])

    img_base = lib.render_raw(env)
    frame_base = lib.preprocess_to_model_input(img_base)
    pix_base = lib.world_to_pixel(env, obj_pos, render_res=64)

    env.sim.set_state(MjSimState(time=tm, qpos=qpos.copy(), qvel=qvel.copy()))
    env.sim.forward()
    lib.set_object_world_pos(env, obj_pos + (epsilon_mm / 1000.0) * dir_vec, obj_index)
    img_pert = lib.render_raw(env)
    frame_pert = lib.preprocess_to_model_input(img_pert)
    pert_pos = lib.get_object_world_pos(env, obj_index)
    pix_pert = lib.world_to_pixel(env, pert_pos, render_res=64)

    return frame_base, frame_pert, pix_base, pix_pert


def adaptive_epsilon_search_at(env, log, t_i, tokenizer, model, context_pixel_values, device,
                                 direction="lateral_x", obj_index=0):
    tried = []
    for eps in EPSILON_CANDIDATES_MM:
        frame_base, frame_pert, pix_base, pix_pert = render_clean_pair_at(env, log, t_i, eps, direction, obj_index)
        h, diff_mask = hamming_only(tokenizer, model, context_pixel_values, frame_base, frame_pert, device)
        pixel_disp = float(np.linalg.norm(pix_pert.astype(np.float64) - pix_base.astype(np.float64)))
        tried.append({"epsilon_mm": eps, "hamming": h, "pixel_disp": pixel_disp,
                       "frame_base": frame_base, "frame_pert": frame_pert, "diff_mask": diff_mask})
        if TARGET_BAND[0] <= h <= TARGET_BAND[1]:
            return {"chosen": tried[-1], "all_tried": tried, "hit_band": True}

    def dist_to_band(h):
        if h < TARGET_BAND[0]:
            return TARGET_BAND[0] - h
        if h > TARGET_BAND[1]:
            return h - TARGET_BAND[1]
        return 0
    best = min(tried, key=lambda d: dist_to_band(d["hamming"]))
    return {"chosen": best, "all_tried": tried, "hit_band": False}
