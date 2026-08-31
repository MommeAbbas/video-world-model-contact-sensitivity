"""
Qualitative example selection for the paper's Figure 3 / qualitative-and-
tracker analysis (Section "Qualitative and Physical Interpretation"). Not a
new experiment: no new hypothesis, metric, offset, perturbation procedure,
episode set, or statistical analysis is introduced. This script only
(1) selects representative already-computed records objectively,
(2) deterministically reconstructs their exact intervention + rollout using
the existing frozen functions, asserting a hard match against the stored
quantitative results before proceeding, and (3) decodes to RGB for figures.
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
from src.model_loading import load_models, pick_device, CONTEXT_LENGTH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
QUAL_DIR = os.path.join(OUT_DIR, "qualitative")
os.makedirs(QUAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: objective selection (no visual cherry-picking). Score = distance to
# group median divergence + 0.5*distance to group median injection_hamming,
# with a hard penalty for hit_target_band=False. Ties broken deterministically
# by (lowest seed, lowest tau) -- never by inspecting the images.
# ---------------------------------------------------------------------------
CATEGORIES = [("onset", -3), ("onset", 2), ("release", -3), ("release", 2)]
N_EXTRAS = 2  # additional near-top candidates per category, kept only to sanity-check the selection


def select_candidates(rows):
    selections = {}
    for event_type, dt in CATEGORIES:
        group = [r for r in rows if r["event_type"] == event_type and r["dt"] == dt]
        vals = np.array([r["mean_downstream_divergence"] for r in group])
        ham_vals = np.array([r["injection_hamming"] for r in group])
        median_div, median_ham = float(np.median(vals)), float(np.median(ham_vals))

        def score(r):
            band_penalty = 0 if r["hit_target_band"] else 100
            return band_penalty + abs(r["mean_downstream_divergence"] - median_div) + 0.5 * abs(r["injection_hamming"] - median_ham)

        ranked = sorted(group, key=lambda r: (round(score(r), 6), r["seed"], r["tau"]))
        selections[f"{event_type}_dt{dt:+d}"] = {
            "event_type": event_type, "dt": dt, "median_div": median_div, "median_ham": median_ham,
            "n_candidates": len(group), "primary": ranked[0], "extras": ranked[1:1 + N_EXTRAS],
        }
    return selections


# ---------------------------------------------------------------------------
# Step 2: deterministic reconstruction with HARD-FAIL verification.
# ---------------------------------------------------------------------------
def reconstruct(env, tokenizer, model, device, episodes, record):
    seed, tau, dt, batch = record["seed"], record["tau"], record["dt"], record["batch"]
    log = episodes[batch][seed]
    t_i = tau + dt
    context_np = log["model_frame_64"][t_i - 2:t_i]
    context = torch.from_numpy(context_np).unsqueeze(0).to(device)

    frame_base, frame_pert, pix_base, pix_pert = render_clean_pair_at(
        env, log, t_i, record["epsilon_mm"], direction="lateral_x")

    hamming, diff_mask = hamming_only(tokenizer, model, context, frame_base, frame_pert, device)
    if hamming != record["injection_hamming"]:
        raise RuntimeError(
            f"RECONSTRUCTION MISMATCH (injection_hamming): stored={record['injection_hamming']} "
            f"reconstructed={hamming} for seed={seed} tau={tau} dt={dt}. STOPPING -- "
            f"do not build any figure from an unverified reconstruction.")

    prefix_A = build_prefix_tokens(tokenizer, model, context, frame_base, device)
    prefix_B = build_prefix_tokens(tokenizer, model, context, frame_pert, device)
    local_actions = log["action"][t_i - 1:t_i + 4].astype(np.float32)
    action_t = torch.from_numpy(local_actions).unsqueeze(0).to(device)

    tokens_A, _ = sl.generate_frames(model, prefix_A.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    tokens_B, _ = sl.generate_frames(model, prefix_B.clone(), action_t, start_frame=1,
                                       end_frame_exclusive=4, do_sample=False)
    div = divergence_curve(model, tokens_A, tokens_B, 1, 4)
    if div != record["downstream_divergence"]:
        raise RuntimeError(
            f"RECONSTRUCTION MISMATCH (downstream_divergence): stored={record['downstream_divergence']} "
            f"reconstructed={div} for seed={seed} tau={tau} dt={dt}. STOPPING -- "
            f"do not build any figure from an unverified reconstruction.")
    print(f"  VERIFIED exact match: seed={seed} tau={tau} dt={dt} batch={batch} "
          f"hamming={hamming} div={div}")

    with torch.no_grad():
        recon_A = tokenizer.detokenize(tokens_A[:, :-1], CONTEXT_LENGTH).clamp(0, 1)[0]  # (6,3,64,64)
        recon_B = tokenizer.detokenize(tokens_B[:, :-1], CONTEXT_LENGTH).clamp(0, 1)[0]

    return {
        "context": context_np,  # (2,3,64,64) raw real frames
        "frame_base": frame_base.numpy(), "frame_pert": frame_pert.numpy(),  # raw rendered A/B, pre-tokenization
        "diff_mask": diff_mask,
        "recon_A_future": recon_A[3:6].cpu().numpy(),  # model-generated +1,+2,+3 for branch A
        "recon_B_future": recon_B[3:6].cpu().numpy(),  # model-generated +1,+2,+3 for branch B
        "recon_A_intervention": recon_A[2].cpu().numpy(),  # sanity cross-check only
        "recon_B_intervention": recon_B[2].cpu().numpy(),
        "hamming": hamming, "downstream_divergence": div, "t_i": t_i,
    }


def to_uint8(chw):
    return (np.clip(chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


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

    print("=" * 70 + "\nSTEP 1: objective selection\n" + "=" * 70)
    selections = select_candidates(rows)
    for key, sel in selections.items():
        r = sel["primary"]
        print(f"\n[{key}] n_candidates={sel['n_candidates']} median_div={sel['median_div']:.2f} "
              f"median_hamming={sel['median_ham']:.1f}")
        print(f"  PRIMARY: seed={r['seed']} batch={r['batch']} tau={r['tau']} hit_band={r['hit_target_band']} "
              f"hamming={r['injection_hamming']} eps={r['epsilon_mm']} div={r['downstream_divergence']} "
              f"mean_div={r['mean_downstream_divergence']:.2f}")
        for e in sel["extras"]:
            print(f"  extra: seed={e['seed']} batch={e['batch']} tau={e['tau']} "
                  f"mean_div={e['mean_downstream_divergence']:.2f}")

    manifest = {k: {"primary": v["primary"], "extras": v["extras"], "median_div": v["median_div"],
                     "median_ham": v["median_ham"], "n_candidates": v["n_candidates"]}
                for k, v in selections.items()}
    with open(os.path.join(QUAL_DIR, "selected_records_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70 + "\nSTEP 2+3: deterministic reconstruction (hard-fail verified) + decode\n" + "=" * 70)
    env = lib.build_env(seed=0)
    reconstructed = {}
    for key, sel in selections.items():
        print(f"\n--- {key} (primary + {len(sel['extras'])} extras) ---")
        reconstructed[key] = {"primary": reconstruct(env, tokenizer, model, device, episodes, sel["primary"])}
        reconstructed[key]["extras"] = [reconstruct(env, tokenizer, model, device, episodes, e)
                                          for e in sel["extras"]]

    # Save individual frames for every primary example
    for key, data in reconstructed.items():
        d = data["primary"]
        prefix = os.path.join(QUAL_DIR, f"{key}")
        np.savez(prefix + "_frames.npz",
                  context=d["context"], frame_base=d["frame_base"], frame_pert=d["frame_pert"],
                  recon_A_future=d["recon_A_future"], recon_B_future=d["recon_B_future"])
    print(f"\nSaved per-example frame arrays to {QUAL_DIR}/*_frames.npz")

    return selections, reconstructed


if __name__ == "__main__":
    main()
