"""
Event-wise bootstrap comparison of natural rollout-error slope_change
against matched free-motion controls (paper Section "Contact-Conditioned
Rollout Dynamics", first paragraph). slope_change = slope_after -
slope_before, computed per event; onset/release are each compared to
control by an independent two-sample bootstrap resampling events with
replacement.

Resamples individual events, not episodes; this does not correct for
multiple events sharing an episode. The episode-cluster-corrected version
is a separate, later analysis (analyze_confound_adjusted_rollout.py /
bootstrap_confound_adjusted_effect.py). Do not add episode clustering here.

Reproduces:
  onset:   diff ~= -0.00189, 95% CI ~= [-0.00706, +0.00330], p ~= 0.4859
  release: diff ~= -0.01192, 95% CI ~= [-0.01790, -0.00624], p ~= 0.0000
"""
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
rel_time = np.arange(-3, 7)


def per_curve_slope_change(curves):
    before_idx = np.where(rel_time < 0)[0]
    after_idx = np.where(rel_time >= 0)[0]
    slope_before = np.diff(curves[:, before_idx], axis=1).mean(axis=1)
    slope_after = np.diff(curves[:, after_idx], axis=1).mean(axis=1)
    return slope_after - slope_before


def bootstrap_diff(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    diffs = np.zeros(n)
    for i in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    p_two_sided = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    ci = np.percentile(diffs, [2.5, 97.5])
    return obs, ci, p_two_sided


def main():
    data = np.load(os.path.join(OUT_DIR, "rq1_pilot_curves.npz"))
    onset, release, control = data["onset"], data["release"], data["control"]

    onset_sc = per_curve_slope_change(onset)
    release_sc = per_curve_slope_change(release)
    control_sc = per_curve_slope_change(control)

    print("mean slope-change (after-before) per event type:")
    print("  onset:  ", onset_sc.mean(), "+/-", onset_sc.std(ddof=1) / np.sqrt(len(onset_sc)))
    print("  release:", release_sc.mean(), "+/-", release_sc.std(ddof=1) / np.sqrt(len(release_sc)))
    print("  control:", control_sc.mean(), "+/-", control_sc.std(ddof=1) / np.sqrt(len(control_sc)))

    obs_on, ci_on, p_on = bootstrap_diff(onset_sc, control_sc)
    print(f"onset vs control: diff={obs_on:.5f} 95%CI={ci_on} p~{p_on:.4f}")
    obs_rel, ci_rel, p_rel = bootstrap_diff(release_sc, control_sc)
    print(f"release vs control: diff={obs_rel:.5f} 95%CI={ci_rel} p~{p_rel:.4f}")

    out = {
        "n_boot": 20000, "rng_seed": 0, "resampling_unit": "event",
        "onset_vs_control": {"diff": float(obs_on), "ci95": ci_on.tolist(), "p": float(p_on)},
        "release_vs_control": {"diff": float(obs_rel), "ci95": ci_rel.tolist(), "p": float(p_rel)},
    }
    out_path = os.path.join(OUT_DIR, "rq1_pilot_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
