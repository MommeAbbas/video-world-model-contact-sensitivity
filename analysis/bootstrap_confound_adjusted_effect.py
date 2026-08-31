"""
Episode-cluster bootstrap for the confound-adjusted regression ("Model 4" in
analyze_confound_adjusted_rollout.py::part_c()):

    slope_change ~ slope_before + release_dummy + onset_dummy
                   + mean_post_vel + mean_post_action

Reuses load_all()/build_table()/ols() from analyze_confound_adjusted_rollout.py
unchanged, refitting inside each of 5000 episode-cluster resamples (resample
unit is episode/seed; a resampled episode's full set of event-window rows is
included every time it is drawn). Writes outputs/rq1_model4_cluster_bootstrap.json,
not touching rq1_confound_results.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_confound_adjusted_rollout import load_all, build_table, ols

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
RNG_SEED = 0
N_BOOT = 5000

MODEL4_X_NAMES = ["slope_before", "release_dummy", "onset_dummy", "mean_post_vel", "mean_post_action"]


def build_model4_arrays(rows):
    slope_after = np.array([r["slope_after"] for r in rows])
    slope_before = np.array([r["slope_before"] for r in rows])
    slope_change = slope_after - slope_before
    etype = np.array([r["event_type"] for r in rows])
    release_dummy = (etype == "release").astype(float)
    onset_dummy = (etype == "onset").astype(float)
    mean_post_vel = np.array([r["mean_post_vel"] for r in rows])
    mean_post_action = np.array([r["mean_post_action"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    y = slope_change
    X_cols = [slope_before, release_dummy, onset_dummy, mean_post_vel, mean_post_action]
    return y, X_cols, seeds


def cluster_bootstrap_p(boot_vals):
    boot_vals = np.asarray(boot_vals)
    frac_ge = (boot_vals >= 0).mean()
    frac_le = (boot_vals <= 0).mean()
    return min(2 * min(frac_ge, frac_le), 1.0)


def main():
    episodes, events, curves = load_all()
    rows = build_table(episodes, events, curves)
    print(f"Loaded {len(rows)} event-window rows (unchanged 96-row dataset).")

    y, X_cols, seeds = build_model4_arrays(rows)
    n = len(y)
    assert n == 96, f"expected n=96, got {n}"

    original = ols(y, X_cols, MODEL4_X_NAMES)
    beta_release_orig = original["coefs"]["release_dummy"]["beta"]
    beta_onset_orig = original["coefs"]["onset_dummy"]["beta"]
    print(f"\nOriginal (unresampled) Model 4 fit, n={original['n']}, R^2={original['r2']:.4f}")
    for k, v in original["coefs"].items():
        print(f"  {k:18s} beta={v['beta']:+.6f}  se={v['se']:.6f}  p(naive)={v['p']:.4f}")
    print(f"\nConfirm point estimates: release_dummy={beta_release_orig:.6f} (expect ~-0.003475), "
          f"onset_dummy={beta_onset_orig:.6f} (expect ~-0.00148)")

    seed_ids = sorted(set(seeds.tolist()))
    by_episode_idx = {s: np.where(seeds == s)[0] for s in seed_ids}
    n_episodes = len(seed_ids)
    print(f"\nEpisode clusters: {n_episodes} -> {seed_ids}")

    rng = np.random.default_rng(RNG_SEED)
    boot_release = []
    boot_onset = []
    n_failed = 0
    failed_reasons = []

    for b in range(N_BOOT):
        sampled_episodes = rng.choice(seed_ids, size=n_episodes, replace=True)
        idx = np.concatenate([by_episode_idx[s] for s in sampled_episodes])
        y_b = y[idx]
        X_cols_b = [col[idx] for col in X_cols]

        X_b = np.column_stack([np.ones(len(y_b))] + X_cols_b)
        rank = np.linalg.matrix_rank(X_b)
        if rank < X_b.shape[1]:
            n_failed += 1
            failed_reasons.append({"replicate": b, "reason": "rank_deficient",
                                     "rank": int(rank), "n_cols": int(X_b.shape[1]),
                                     "n_rows": int(len(y_b))})
            continue
        try:
            fit_b = ols(y_b, X_cols_b, MODEL4_X_NAMES)
        except np.linalg.LinAlgError as e:
            n_failed += 1
            failed_reasons.append({"replicate": b, "reason": f"LinAlgError: {e}"})
            continue

        boot_release.append(fit_b["coefs"]["release_dummy"]["beta"])
        boot_onset.append(fit_b["coefs"]["onset_dummy"]["beta"])

    n_valid = len(boot_release)
    print(f"\nBootstrap replicates: {N_BOOT} requested, {n_valid} valid, {n_failed} failed/rank-deficient")

    boot_release = np.array(boot_release)
    boot_onset = np.array(boot_onset)

    def summarize(orig, boot):
        ci = np.percentile(boot, [2.5, 97.5])
        p = cluster_bootstrap_p(boot)
        return {
            "original_beta": float(orig),
            "bootstrap_mean": float(boot.mean()),
            "bootstrap_median": float(np.median(boot)),
            "ci95_lo": float(ci[0]), "ci95_hi": float(ci[1]),
            "cluster_bootstrap_p": float(p),
            "n_valid_replicates": int(len(boot)),
        }

    release_summary = summarize(beta_release_orig, boot_release)
    onset_summary = summarize(beta_onset_orig, boot_onset)

    print(f"\nrelease_dummy:")
    print(f"  original beta      = {release_summary['original_beta']:+.6f}")
    print(f"  bootstrap mean      = {release_summary['bootstrap_mean']:+.6f}")
    print(f"  bootstrap median    = {release_summary['bootstrap_median']:+.6f}")
    print(f"  95% CI              = [{release_summary['ci95_lo']:+.6f}, {release_summary['ci95_hi']:+.6f}]")
    print(f"  cluster-bootstrap p = {release_summary['cluster_bootstrap_p']:.4f}")
    inside = release_summary['ci95_lo'] <= release_summary['original_beta'] <= release_summary['ci95_hi']
    print(f"  original beta inside CI: {inside}")

    print(f"\nonset_dummy:")
    print(f"  original beta      = {onset_summary['original_beta']:+.6f}")
    print(f"  bootstrap mean      = {onset_summary['bootstrap_mean']:+.6f}")
    print(f"  bootstrap median    = {onset_summary['bootstrap_median']:+.6f}")
    print(f"  95% CI              = [{onset_summary['ci95_lo']:+.6f}, {onset_summary['ci95_hi']:+.6f}]")
    print(f"  cluster-bootstrap p = {onset_summary['cluster_bootstrap_p']:.4f}")
    inside = onset_summary['ci95_lo'] <= onset_summary['original_beta'] <= onset_summary['ci95_hi']
    print(f"  original beta inside CI: {inside}")

    out = {
        "model": "slope_change ~ slope_before + release_dummy + onset_dummy + mean_post_vel + mean_post_action",
        "n_rows": int(n),
        "n_episodes": int(n_episodes),
        "episode_ids": [int(s) for s in seed_ids],
        "rng_seed": RNG_SEED,
        "n_boot_requested": N_BOOT,
        "n_boot_valid": n_valid,
        "n_boot_failed": n_failed,
        "failed_replicate_details": failed_reasons,
        "original_fit_all_coefs": original["coefs"],
        "release_dummy": release_summary,
        "onset_dummy": onset_summary,
    }
    out_path = os.path.join(OUT_DIR, "rq1_model4_cluster_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
