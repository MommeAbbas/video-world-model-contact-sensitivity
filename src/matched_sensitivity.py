"""
Shared infrastructure for the matched predictive-sensitivity experiment
(paper Section "Causal Sensitivity Around Contact Transitions"): adaptive
per-event epsilon search targeting a non-saturating injection-Hamming band,
covariate computation for nearest-neighbor control matching, free-motion
control-candidate enumeration, greedy matching, and the per-event
intervention + rollout + divergence measurement used for every onset/release
experiment (exploratory, confirmation, and release batches alike).
"""
import numpy as np
import torch

from . import causal_splice as sl
from . import simulator_interface as lib
from .causal_intervention import build_segment, build_prefix_tokens, divergence_curve
from .model_loading import CONTEXT_LENGTH
from robosuite.utils.binding_utils import MjSimState

EPSILON_CANDIDATES_MM = [0.5, 1.0, 2.0, 3.0, 5.0]
TARGET_BAND = (2, 5)  # inclusive, "roughly 2-5 changed dyna tokens"
N_CONTINUE = 3


def render_clean_pair(env, log, t_event, epsilon_mm, direction="lateral_x"):
    t_int_abs = t_event - 3
    qpos, qvel, tm = log["qpos"][t_int_abs], log["qvel"][t_int_abs], log["time"][t_int_abs]

    env.sim.set_state(MjSimState(time=tm, qpos=qpos.copy(), qvel=qvel.copy()))
    env.sim.forward()
    cube_pos = lib.get_object_world_pos(env, 0)
    axes = lib.camera_axes_world(env)
    dir_vec = axes[direction] / np.linalg.norm(axes[direction])

    img_base = lib.render_raw(env)
    frame_base = lib.preprocess_to_model_input(img_base)
    pix_base = lib.world_to_pixel(env, cube_pos, render_res=64)

    env.sim.set_state(MjSimState(time=tm, qpos=qpos.copy(), qvel=qvel.copy()))
    env.sim.forward()
    lib.set_object_world_pos(env, cube_pos + (epsilon_mm / 1000.0) * dir_vec, 0)
    img_pert = lib.render_raw(env)
    frame_pert = lib.preprocess_to_model_input(img_pert)
    pert_pos = lib.get_object_world_pos(env, 0)
    pix_pert = lib.world_to_pixel(env, pert_pos, render_res=64)

    return frame_base, frame_pert, pix_base, pix_pert


@torch.no_grad()
def hamming_only(tokenizer, model, context_pixel_values, frame_base, frame_pert, device):
    dyna_base = sl.retokenize_single_frame(tokenizer, context_pixel_values, frame_base.unsqueeze(0).to(device),
                                             CONTEXT_LENGTH)
    dyna_pert = sl.retokenize_single_frame(tokenizer, context_pixel_values, frame_pert.unsqueeze(0).to(device),
                                             CONTEXT_LENGTH)
    diff = (dyna_base.to(dyna_pert.dtype) != dyna_pert)
    return int(diff.sum().item()), diff[0].cpu().numpy()


def adaptive_epsilon_search(env, log, t_event, tokenizer, model, context_pixel_values, device,
                              direction="lateral_x"):
    """Tries EPSILON_CANDIDATES_MM in order, returns the first landing inside
    TARGET_BAND; if none land inside, returns the candidate whose Hamming is
    closest to the band. Returns a dict with all diagnostics.
    """
    tried = []
    for eps in EPSILON_CANDIDATES_MM:
        frame_base, frame_pert, pix_base, pix_pert = render_clean_pair(env, log, t_event, eps, direction)
        h, diff_mask = hamming_only(tokenizer, model, context_pixel_values, frame_base, frame_pert, device)
        pixel_disp = float(np.linalg.norm(pix_pert.astype(np.float64) - pix_base.astype(np.float64)))
        tried.append({"epsilon_mm": eps, "hamming": h, "pixel_disp": pixel_disp,
                       "frame_base": frame_base, "frame_pert": frame_pert, "diff_mask": diff_mask})
        if TARGET_BAND[0] <= h <= TARGET_BAND[1]:
            return {"chosen": tried[-1], "all_tried": tried, "hit_band": True}
    # none landed in band: pick closest
    def dist_to_band(h):
        if h < TARGET_BAND[0]:
            return TARGET_BAND[0] - h
        if h > TARGET_BAND[1]:
            return h - TARGET_BAND[1]
        return 0
    best = min(tried, key=lambda d: dist_to_band(d["hamming"]))
    return {"chosen": best, "all_tried": tried, "hit_band": False}


def window_covariates(log, t_event, env):
    rel_lo, rel_hi = -5, 6
    start, end = t_event + rel_lo, t_event + rel_hi
    vel = log["obj_velocity"][start:end + 1, 0]
    act = log["action_norm"][start:end + 1]
    t_int_abs = t_event - 3
    cube_pos = log["obj_pos"][t_int_abs, 0]
    eef_pos = log["eef_pos"][t_int_abs]
    cam_pos = env.sim.data.cam_xpos[env.sim.model.camera_name2id(lib.CAMERA_NAME)]
    depth = float(np.linalg.norm(cube_pos - cam_pos))
    gripper_obj_dist = float(np.linalg.norm(cube_pos - eef_pos))
    return {
        "mean_action_norm": float(act.mean()),
        "mean_obj_velocity": float(vel.mean()),
        "object_depth": depth,
        "gripper_object_distance": gripper_obj_dist,
    }


def all_control_candidates(episodes):
    """All (seed,t) with no cube contact transition anywhere in the 12-step
    window, for nearest-neighbor control matching.
    """
    candidates = []
    for seed, log in episodes.items():
        col = log["contact_per_object"][:, 0].astype(bool)
        T = len(col)
        for t in range(5, T - 6):
            window = col[t - 5:t + 7]
            if window.any() == window.all():
                candidates.append((seed, t))
    return candidates


def match_controls(onset_events, control_pool, episodes, env):
    """Greedy nearest-neighbor matching without replacement on standardized
    [mean_action_norm, mean_obj_velocity, object_depth].
    """
    onset_cov = [window_covariates(episodes[s], t, env) for s, t in onset_events]
    pool_cov = [window_covariates(episodes[s], t, env) for s, t in control_pool]

    keys = ["mean_action_norm", "mean_obj_velocity", "object_depth"]
    pool_mat = np.array([[c[k] for k in keys] for c in pool_cov])
    onset_mat = np.array([[c[k] for k in keys] for c in onset_cov])
    mu, sigma = pool_mat.mean(axis=0), pool_mat.std(axis=0) + 1e-8
    pool_std = (pool_mat - mu) / sigma
    onset_std = (onset_mat - mu) / sigma

    used = set()
    matches = []
    for i, ov in enumerate(onset_std):
        dists = np.linalg.norm(pool_std - ov, axis=1)
        order = np.argsort(dists)
        chosen = None
        for j in order:
            if j not in used:
                chosen = j
                break
        if chosen is None:
            chosen = order[0]  # fall back to reuse if pool exhausted
        used.add(chosen)
        matches.append((control_pool[chosen], dists[chosen], onset_cov[i], pool_cov[chosen]))
    return matches


@torch.no_grad()
def run_event(env, tokenizer, model, device, log, t_event, event_type, seed):
    built = build_segment(log, t_event)
    if built is None:
        return None
    context, model_actions, gt_future = built
    context_pixel_values = torch.from_numpy(context).unsqueeze(0).to(device)
    action_t = torch.from_numpy(model_actions).unsqueeze(0).to(device)

    search = adaptive_epsilon_search(env, log, t_event, tokenizer, model, context_pixel_values, device)
    chosen = search["chosen"]

    prefix_A = build_prefix_tokens(tokenizer, model, context_pixel_values, chosen["frame_base"], device)
    prefix_B = build_prefix_tokens(tokenizer, model, context_pixel_values, chosen["frame_pert"], device)
    s, e = sl.dyna_slice(model, 0)
    hamming0 = (prefix_A[:, s:e] != prefix_B[:, s:e]).sum().item()
    changed_idx = torch.where((prefix_A[:, s:e] != prefix_B[:, s:e])[0])[0].cpu().numpy().tolist()

    end_frame = 1 + N_CONTINUE
    tokens_A, _ = sl.generate_frames(model, prefix_A.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=end_frame, do_sample=False)
    tokens_B, _ = sl.generate_frames(model, prefix_B.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=end_frame, do_sample=False)
    div_curve = divergence_curve(model, tokens_A, tokens_B, 1, end_frame)

    cov = window_covariates(log, t_event, env)

    return {
        "event_type": event_type, "seed": int(seed), "t": int(t_event),
        "epsilon_mm": chosen["epsilon_mm"], "direction": "lateral_x",
        "hit_target_band": search["hit_band"],
        "image_displacement_px": chosen["pixel_disp"],
        "injection_hamming": hamming0, "changed_token_indices": changed_idx,
        "epsilon_search_trace": [{"epsilon_mm": d["epsilon_mm"], "hamming": d["hamming"]} for d in search["all_tried"]],
        **cov,
        "downstream_divergence": div_curve,
        "mean_downstream_divergence": float(np.mean(div_curve)),
        "cumulative_downstream_divergence": float(np.sum(div_curve)),
    }
