"""
Episode-cluster bootstrap for Model 2, the covariate-adjusted regression
from analyze_matched_sensitivity.py (contact_status + injection_hamming +
image_displacement + action_magnitude + object_velocity + object_depth +
gripper_distance): refits the specification inside every episode-level
resample, as Model 1's beta_contact already is inside analyze(). This is
the correctly-clustered uncertainty behind the paper's robustness claim that
the effect survives adjustment for image displacement and gripper-object
distance: beta ~= +2.17, 95% cluster CI ~= [+1.14, +3.63], p ~= 0.0007
(combined batch).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matched_sensitivity import load, ols

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def bootstrap_model2(rows, treatment_label, n_boot=3000, seed=0):
    contact = np.array([1.0 if r["event_type"] == treatment_label else 0.0 for r in rows])
    hamming = np.array([r["injection_hamming"] for r in rows])
    div = np.array([r["mean_downstream_divergence"] for r in rows])
    img_disp = np.array([r["image_displacement_px"] for r in rows])
    action_mag = np.array([r["mean_action_norm"] for r in rows])
    obj_vel = np.array([r["mean_obj_velocity"] for r in rows])
    depth = np.array([r["object_depth"] for r in rows])
    visibility = np.array([r["gripper_object_distance"] for r in rows])
    pair_id = np.array([r["pair_id"] for r in rows])

    treat_rows = [r for r in rows if r["event_type"] == treatment_label]
    pair_to_seed = {r["pair_id"]: r["seed"] for r in treat_rows}
    clusters = sorted(set(pair_to_seed.values()))

    def fit(idx):
        return ols(div[idx], [contact[idx], hamming[idx], img_disp[idx], action_mag[idx],
                                obj_vel[idx], depth[idx], visibility[idx]],
                   ["contact_status", "injection_hamming", "image_displacement", "action_magnitude",
                    "object_velocity", "object_depth", "visibility(gripper_dist)"])

    point = fit(np.arange(len(rows)))["coefs"]["contact_status"]["beta"]

    rng = np.random.default_rng(seed)
    betas = []
    n_fail = 0
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sel_pairs = []
        for c in sampled_clusters:
            sel_pairs.extend([pid for pid, s in pair_to_seed.items() if s == c])
        idx = np.array([i for i, r in enumerate(rows) if r["pair_id"] in sel_pairs])
        if len(idx) < 10 or contact[idx].std() == 0:
            n_fail += 1
            continue
        try:
            betas.append(fit(idx)["coefs"]["contact_status"]["beta"])
        except Exception:
            n_fail += 1
            continue
    betas = np.array(betas)
    ci = np.percentile(betas, [2.5, 97.5]) if len(betas) > 10 else (np.nan, np.nan)
    p = 2 * min((betas >= 0).mean(), (betas <= 0).mean()) if len(betas) > 10 else np.nan
    return {
        "n_clusters": len(clusters), "n_valid_draws": len(betas), "n_failed_draws": n_fail,
        "point_estimate": float(point), "cluster_ci95": [float(ci[0]), float(ci[1])],
        "bootstrap_p": float(p),
    }


def main():
    print("Episode-cluster bootstrap for Model 2 (release, refit inside resamples)")
    for name, label in [
        ("rq2_release_exploratory_full.json", "exploratory (seeds 0-9)"),
        ("rq2_release_confirmation_full.json", "confirmation (seeds 100-109)"),
        ("rq2_release_combined_full.json", "combined"),
    ]:
        rows = load(name)
        result = bootstrap_model2(rows, treatment_label="release")
        print(f"\n[{label}] n_events={len(rows)}")
        print(f"  Model 2 beta_release point estimate = {result['point_estimate']:+.4f}")
        print(f"  n clusters(episodes) = {result['n_clusters']}, "
              f"valid bootstrap draws = {result['n_valid_draws']} (failed={result['n_failed_draws']})")
        print(f"  episode-cluster 95% CI = [{result['cluster_ci95'][0]:+.4f}, {result['cluster_ci95'][1]:+.4f}]")
        print(f"  bootstrap two-sided p ~ {result['bootstrap_p']:.4f}")


if __name__ == "__main__":
    main()
