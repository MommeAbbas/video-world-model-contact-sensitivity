"""
Primary analysis for the matched predictive-sensitivity experiment (paper
Table 1): Model 1 (contact_status + injection_hamming), Model 2 (the
covariate-adjusted robustness check, adding image displacement, action
magnitude, object velocity/depth, gripper distance), an episode-cluster
bootstrap on Model 1's beta_contact, and a matched-pair divergence-difference
analysis. Parameterized by treatment_label so the same implementation
serves both onset and release.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def ols(y, X_cols, X_names):
    n = len(y)
    X = np.column_stack([np.ones(n)] + X_cols)
    names = ["intercept"] + X_names
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))
    ci95 = np.column_stack([beta - 1.96 * se, beta + 1.96 * se])
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {
        "n": n, "r2": float(r2), "dof": int(dof),
        "coefs": {name: {"beta": float(b), "se": float(s), "t": float(tv), "p": float(p),
                          "ci95": [float(ci95[i, 0]), float(ci95[i, 1])]}
                  for i, (name, b, s, tv, p) in enumerate(zip(names, beta, se, tvals, pvals))},
    }


def load(name="rq2_scaled_full.json"):
    with open(os.path.join(OUT_DIR, name)) as f:
        rows = json.load(f)
    return rows


def analyze(rows, label="", treatment_label="onset"):
    # treatment_label generalizes this to run on "release"-labeled rows too;
    # the default reproduces the onset-only behavior.
    print(f"\nAnalysis: {label} (n_events={len(rows)})")

    contact = np.array([1.0 if r["event_type"] == treatment_label else 0.0 for r in rows])
    hamming = np.array([r["injection_hamming"] for r in rows])
    div = np.array([r["mean_downstream_divergence"] for r in rows])
    img_disp = np.array([r["image_displacement_px"] for r in rows])
    action_mag = np.array([r["mean_action_norm"] for r in rows])
    obj_vel = np.array([r["mean_obj_velocity"] for r in rows])
    depth = np.array([r["object_depth"] for r in rows])
    visibility = np.array([r["gripper_object_distance"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    pair_id = np.array([r["pair_id"] for r in rows])

    print(f"\nInjection Hamming distribution: {treatment_label} mean={hamming[contact==1].mean():.2f} "
          f"(sd={hamming[contact==1].std():.2f}), control mean={hamming[contact==0].mean():.2f} "
          f"(sd={hamming[contact==0].std():.2f})")
    print(f"Covariate balance ({treatment_label} vs control means): "
          f"action_norm {action_mag[contact==1].mean():.3f} vs {action_mag[contact==0].mean():.3f}, "
          f"obj_velocity {obj_vel[contact==1].mean():.4f} vs {obj_vel[contact==0].mean():.4f}, "
          f"depth {depth[contact==1].mean():.3f} vs {depth[contact==0].mean():.3f}, "
          f"gripper_obj_dist {visibility[contact==1].mean():.3f} vs {visibility[contact==0].mean():.3f}")

    print("\nModel 1: future_divergence ~ contact_status + injection_hamming")
    m1 = ols(div, [contact, hamming], ["contact_status", "injection_hamming"])
    for k, v in m1["coefs"].items():
        print(f"  {k:16s} beta={v['beta']:+.4f} se={v['se']:.4f} 95%CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.4f}")
    print(f"  R^2={m1['r2']:.3f}")

    print("\nModel 2: + image_displacement + action_magnitude + object_velocity + object_depth + visibility")
    m2 = ols(div, [contact, hamming, img_disp, action_mag, obj_vel, depth, visibility],
             ["contact_status", "injection_hamming", "image_displacement", "action_magnitude",
              "object_velocity", "object_depth", "visibility(gripper_dist)"])
    for k, v in m2["coefs"].items():
        print(f"  {k:26s} beta={v['beta']:+.4f} se={v['se']:.4f} 95%CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.4f}")
    print(f"  R^2={m2['r2']:.3f}")

    # Cluster by the treatment episode of each pair, the primary non-independence source.
    print(f"\nEpisode-cluster bootstrap on beta_contact (Model 1), clustering by pair's {treatment_label} episode:")
    onset_rows = [r for r in rows if r["event_type"] == treatment_label]
    pair_to_onset_seed = {r["pair_id"]: r["seed"] for r in onset_rows}
    clusters = sorted(set(pair_to_onset_seed.values()))
    rng = np.random.default_rng(0)
    n_boot = 3000
    betas = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sel_pairs = []
        for c in sampled_clusters:
            sel_pairs.extend([pid for pid, s in pair_to_onset_seed.items() if s == c])
        idx = np.array([i for i, r in enumerate(rows) if r["pair_id"] in sel_pairs])
        if len(idx) < 4 or contact[idx].std() == 0:
            continue
        try:
            m = ols(div[idx], [contact[idx], hamming[idx]], ["contact_status", "injection_hamming"])
            betas.append(m["coefs"]["contact_status"]["beta"])
        except Exception:
            continue
    betas = np.array(betas)
    ci = np.percentile(betas, [2.5, 97.5])
    print(f"  n episodes(clusters)={len(clusters)}, n valid bootstrap draws={len(betas)}")
    print(f"  beta_contact point estimate={m1['coefs']['contact_status']['beta']:+.4f}, "
          f"cluster-bootstrap 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # Matched-pair analysis
    print(f"\nMatched-pair Delta D = D_{treatment_label} - D_control:")
    n_pairs = len(set(pair_id))
    deltas = []
    for pid in sorted(set(pair_id)):
        pr = [r for r in rows if r["pair_id"] == pid]
        d_on = [r["mean_downstream_divergence"] for r in pr if r["event_type"] == treatment_label][0]
        d_ct = [r["mean_downstream_divergence"] for r in pr if r["event_type"] == "control"][0]
        onset_seed = [r["seed"] for r in pr if r["event_type"] == treatment_label][0]
        deltas.append((pid, onset_seed, d_on - d_ct))
    delta_vals = np.array([d[2] for d in deltas])
    print(f"  n pairs={len(deltas)}, mean Delta D={delta_vals.mean():+.3f}, median={np.median(delta_vals):+.3f}, "
          f"sd={delta_vals.std():.3f}")

    # cluster (by onset episode) bootstrap CI on mean Delta D
    seed_of_pair = {d[0]: d[1] for d in deltas}
    delta_of_pair = {d[0]: d[2] for d in deltas}
    pair_clusters = sorted(set(seed_of_pair.values()))
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_clusters, size=len(pair_clusters), replace=True)
        vals = []
        for c in sampled:
            vals.extend([delta_of_pair[pid] for pid, s in seed_of_pair.items() if s == c])
        if vals:
            boot_means.append(np.mean(vals))
    boot_means = np.array(boot_means)
    ci_delta = np.percentile(boot_means, [2.5, 97.5])
    p_delta = 2 * min((boot_means >= 0).mean(), (boot_means <= 0).mean())
    print(f"  cluster-bootstrap 95% CI on mean Delta D=[{ci_delta[0]:+.3f}, {ci_delta[1]:+.3f}], p~{p_delta:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(hamming[contact == 1], bins=range(0, 8), alpha=0.6, label=treatment_label, color="tab:red")
    axes[0].hist(hamming[contact == 0], bins=range(0, 8), alpha=0.6, label="control", color="tab:gray")
    axes[0].set_xlabel("injection Hamming distance"); axes[0].set_ylabel("count")
    axes[0].set_title("Injection Hamming by group"); axes[0].legend()

    steps = np.array([1, 2, 3])
    onset_curve = np.array([r["downstream_divergence"] for r in rows if r["event_type"] == treatment_label])
    control_curve = np.array([r["downstream_divergence"] for r in rows if r["event_type"] == "control"])
    axes[1].errorbar(steps, onset_curve.mean(axis=0), yerr=onset_curve.std(axis=0) / np.sqrt(len(onset_curve)),
                       marker="o", label=treatment_label, color="tab:red")
    axes[1].errorbar(steps, control_curve.mean(axis=0), yerr=control_curve.std(axis=0) / np.sqrt(len(control_curve)),
                       marker="o", label="control", color="tab:gray")
    axes[1].set_xlabel("downstream step"); axes[1].set_ylabel("token Hamming divergence")
    axes[1].set_title("Downstream divergence curves"); axes[1].legend()

    axes[2].hist(delta_vals, bins=15, color="tab:purple", alpha=0.8)
    axes[2].axvline(0, color="k", lw=1)
    axes[2].axvline(delta_vals.mean(), color="tab:purple", lw=2, ls="--")
    axes[2].set_xlabel("Delta D (contact - control)"); axes[2].set_ylabel("count")
    axes[2].set_title(f"Matched-pair Delta D (mean={delta_vals.mean():+.2f})")

    plt.tight_layout()
    import re
    suffix = "_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    plt.savefig(os.path.join(OUT_DIR, f"rq2_scaled_plots{suffix}.png"), dpi=130)
    print(f"\nSaved plots to outputs/rq2_scaled_plots{suffix}.png")

    return {
        "model1": m1, "model2": m2,
        "beta_contact_cluster_ci": ci.tolist(),
        "delta_D_mean": float(delta_vals.mean()), "delta_D_median": float(np.median(delta_vals)),
        "delta_D_cluster_ci": ci_delta.tolist(), "delta_D_p": float(p_delta),
        "n_pairs": n_pairs,
    }


def main():
    if os.path.exists(os.path.join(OUT_DIR, "rq2_scaled_interim20.json")):
        rows20 = load("rq2_scaled_interim20.json")
        analyze(rows20, label="interim (first 20 pairs)")
    rows_full = load("rq2_scaled_full.json")
    result = analyze(rows_full, label="full")
    with open(os.path.join(OUT_DIR, "rq2_scaled_analysis_results.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
