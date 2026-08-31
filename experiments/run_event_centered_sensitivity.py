"""
Event-centered temporal sensitivity sweep (paper Section "Temporal Structure
of Predictive Sensitivity", Figure 2). Runs the simulator-clean intervention +
3-frame continuation at t_i = tau + Delta_t for every eligible,
contamination-clean (event_type, seed, tau, Delta_t) record from
analysis/audit_event_centered_eligibility.py's output. No matched controls
(each event is its own alignment anchor across offsets). No new episodes.
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
from src.event_centered_intervention import adaptive_epsilon_search_at
from src.causal_intervention import build_prefix_tokens, divergence_curve
from src.model_loading import load_models, pick_device

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


@torch.no_grad()
def run_one(env, tokenizer, model, device, log, tau, dt, obj_index=0):
    t_i = tau + dt
    context_np = log["model_frame_64"][t_i - 2:t_i]
    context = torch.from_numpy(context_np).unsqueeze(0).to(device)

    search = adaptive_epsilon_search_at(env, log, t_i, tokenizer, model, context, device, obj_index=obj_index)
    chosen = search["chosen"]

    prefix_A = build_prefix_tokens(tokenizer, model, context, chosen["frame_base"], device)
    prefix_B = build_prefix_tokens(tokenizer, model, context, chosen["frame_pert"], device)
    local_actions = log["action"][t_i - 1:t_i + 4].astype(np.float32)
    action_t = torch.from_numpy(local_actions).unsqueeze(0).to(device)

    tokens_A, _ = sl.generate_frames(model, prefix_A.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    tokens_B, _ = sl.generate_frames(model, prefix_B.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    div = divergence_curve(model, tokens_A, tokens_B, 1, 4)

    return {
        "t_i": int(t_i), "epsilon_mm": chosen["epsilon_mm"], "hit_target_band": search["hit_band"],
        "injection_hamming": chosen["hamming"], "image_displacement_px": chosen["pixel_disp"],
        "changed_token_indices": np.where(chosen["diff_mask"])[0].tolist(),
        "downstream_divergence": div, "mean_downstream_divergence": float(np.mean(div)),
    }


def main():
    device = pick_device()
    print(f"Using device: {device}")
    tokenizer, model = load_models(device)

    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        exploratory = pickle.load(f)
    with open(os.path.join(OUT_DIR, "pilot_episodes_confirmation.pkl"), "rb") as f:
        confirmation = pickle.load(f)
    episodes = {"exploratory": exploratory, "confirmation": confirmation}

    with open(os.path.join(OUT_DIR, "rq2_event_centered_audit.json")) as f:
        audit = json.load(f)
    clean_records = [r for r in audit if r["clean"]]
    print(f"Total clean (event,offset) records to run: {len(clean_records)}")

    env = lib.build_env(seed=0)
    results = []
    for i, rec in enumerate(clean_records):
        log = episodes[rec["batch"]][rec["seed"]]
        out = run_one(env, tokenizer, model, device, log, rec["tau"], rec["dt"])
        row = {**rec, **out}
        results.append(row)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"[{i+1}/{len(clean_records)}] {rec['event_type']} batch={rec['batch']} seed={rec['seed']} "
                  f"tau={rec['tau']} dt={rec['dt']} -> hamming={out['injection_hamming']} "
                  f"div={out['downstream_divergence']}")
        if (i + 1) % 100 == 0:
            with open(os.path.join(OUT_DIR, "rq2_event_centered_results_partial.json"), "w") as f:
                json.dump(results, f, indent=2)

    with open(os.path.join(OUT_DIR, "rq2_event_centered_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} records to outputs/rq2_event_centered_results.json")


if __name__ == "__main__":
    main()
