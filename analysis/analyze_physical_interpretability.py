"""
Physical-interpretability analysis (paper Section "Qualitative and Physical
Interpretation"): applies the validated cube tracker
(src/cube_tracker.py, validated in validation/validate_cube_tracker.py) to a
deterministic, non-cherry-picked sample of the already-computed
event-centered reconstructions, re-derived using the exact stored epsilon
(no re-search) and the exact frozen reconstruction functions -- the same
procedure verified in select_qualitative_examples.py, just also capturing
decoded pixels for tracking. Not a new experiment: no metric, offset,
perturbation, or episode selection is changed.

Sample: up to N_PER_CELL records per (event_type, dt) cell, taken in sorted
(seed, tau) order (deterministic, not selected by outcome).
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import simulator_interface as lib
from src import causal_splice as sl
from src.event_centered_intervention import render_clean_pair_at
from src.matched_sensitivity import hamming_only
from src.causal_intervention import build_prefix_tokens, divergence_curve
from src.cube_tracker import cube_centroid
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
QUAL_DIR = os.path.join(OUT_DIR, "qualitative")
N_PER_CELL = 10
JUMP_FLAG_PX = 20.0  # single-step centroid jump implausible for this scene/horizon -> flag, don't silently average


@torch.no_grad()
def reconstruct_and_track(env, tokenizer, model, device, log, tau, dt, epsilon_mm):
    t_i = tau + dt
    context_np = log["model_frame_64"][t_i - 2:t_i]
    context = torch.from_numpy(context_np).unsqueeze(0).to(device)

    frame_base, frame_pert, _, _ = render_clean_pair_at(env, log, t_i, epsilon_mm, direction="lateral_x")
    hamming, _ = hamming_only(tokenizer, model, context, frame_base, frame_pert, device)

    prefix_A = build_prefix_tokens(tokenizer, model, context, frame_base, device)
    prefix_B = build_prefix_tokens(tokenizer, model, context, frame_pert, device)
    local_actions = log["action"][t_i - 1:t_i + 4].astype(np.float32)
    action_t = torch.from_numpy(local_actions).unsqueeze(0).to(device)

    tokens_A, _ = sl.generate_frames(model, prefix_A.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    tokens_B, _ = sl.generate_frames(model, prefix_B.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    div = divergence_curve(model, tokens_A, tokens_B, 1, 4)

    recon_A = tokenizer.detokenize(tokens_A[:, :-1], CONTEXT_LENGTH).clamp(0, 1)[0]
    recon_B = tokenizer.detokenize(tokens_B[:, :-1], CONTEXT_LENGTH).clamp(0, 1)[0]
    fut_A = recon_A[3:6].cpu().numpy()
    fut_B = recon_B[3:6].cpu().numpy()

    cent_base = cube_centroid(frame_base.numpy())
    cent_pert = cube_centroid(frame_pert.numpy())
    cent_A = [cube_centroid(fut_A[k]) for k in range(3)]
    cent_B = [cube_centroid(fut_B[k]) for k in range(3)]

    return {
        "hamming": hamming, "downstream_divergence": div,
        "cent_base": cent_base, "cent_pert": cent_pert, "cent_A": cent_A, "cent_B": cent_B,
    }


def pixel_dist(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def flag_implausible_track(centroids):
    """centroids: list of (row,col) or None across a branch's rollout
    (intervention + 3 future). Flags if any step-to-step jump exceeds
    JUMP_FLAG_PX or if any step failed to detect."""
    if any(c is None for c in centroids):
        return True
    for i in range(len(centroids) - 1):
        if pixel_dist(centroids[i], centroids[i + 1]) > JUMP_FLAG_PX:
            return True
    return False


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        exploratory = pickle.load(f)
    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "rb") as f:
        confirmation = pickle.load(f)
    episodes = {"exploratory": exploratory, "confirmation": confirmation}

    with open(os.path.join(OUT_DIR, "rq2_event_centered_results.json")) as f:
        rows = json.load(f)

    env = lib.build_env(seed=0)

    sample = []
    for event_type in ["onset", "release"]:
        for dt in [-3, -2, -1, 0, 1, 2, 3]:
            group = sorted([r for r in rows if r["event_type"] == event_type and r["dt"] == dt],
                            key=lambda r: (r["seed"], r["tau"]))
            sample.extend(group[:N_PER_CELL])
    print(f"Deterministic sample size: {len(sample)} records (up to {N_PER_CELL} per (event_type,dt) cell, "
          f"sorted by seed/tau, not selected by outcome)")

    results = []
    for i, rec in enumerate(sample):
        log = episodes[rec["batch"]][rec["seed"]]
        out = reconstruct_and_track(env, tokenizer, model, device, log, rec["tau"], rec["dt"], rec["epsilon_mm"])
        assert out["hamming"] == rec["injection_hamming"], (
            f"MISMATCH hamming: stored={rec['injection_hamming']} got={out['hamming']} for {rec}")
        assert out["downstream_divergence"] == rec["downstream_divergence"], (
            f"MISMATCH divergence: stored={rec['downstream_divergence']} got={out['downstream_divergence']} for {rec}")

        track_A = [out["cent_base"]] + out["cent_A"]
        track_B = [out["cent_pert"]] + out["cent_B"]
        flagged = flag_implausible_track(track_A) or flag_implausible_track(track_B)
        sep_per_step = [pixel_dist(track_A[k], track_B[k]) for k in range(4)]  # intervention,+1,+2,+3
        mean_sep_future = None
        if not flagged and all(s is not None for s in sep_per_step[1:]):
            mean_sep_future = float(np.mean(sep_per_step[1:]))

        results.append({
            **{k: rec[k] for k in ["event_type", "seed", "tau", "dt", "batch", "epsilon_mm"]},
            "injection_hamming": rec["injection_hamming"], "downstream_divergence": rec["downstream_divergence"],
            "mean_downstream_divergence": rec["mean_downstream_divergence"],
            "centroid_sep_intervention": sep_per_step[0], "centroid_sep_future": sep_per_step[1:],
            "mean_centroid_sep_future": mean_sep_future, "flagged": flagged,
        })
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(sample)}] {rec['event_type']} dt={rec['dt']} seed={rec['seed']} tau={rec['tau']} "
                  f"hamming={out['hamming']} mean_sep_future={mean_sep_future}")

    with open(os.path.join(QUAL_DIR, "physical_interpretability_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    n_flagged = sum(r["flagged"] for r in results)
    print(f"\nTotal records: {len(results)}, flagged (tracker failure or implausible jump): {n_flagged} "
          f"({100*n_flagged/len(results):.1f}%)")
    print(f"Saved to {QUAL_DIR}/physical_interpretability_results.json")


if __name__ == "__main__":
    main()
