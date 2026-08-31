"""
Confound-controlled analysis of the natural rollout-error experiment (paper
Section "Contact-Conditioned Rollout Dynamics", second paragraph): controls
for the ceiling/mean-reversion confound (windows that already grew error fast
pre-event, or already sat at a high error level at t=0, mechanically have less
room to keep growing afterward) and for the post-event motion/action
confound (release coincides with a drop in object velocity and action
magnitude).

Part A: ceiling/saturation hypothesis (slope_before vs slope_after
         correlation, error_at_t0 vs slope_after, OLS controlling for
         slope_before).
Part B: episode-cluster bootstrap on the raw release/onset-vs-control
         difference in slope_change (fixes pseudoreplication in a naive
         event-wise bootstrap: multiple events come from the same episode).
Part C: motion/action confound (velocity/action aligned plots + regression).
         "Model 4" here -- slope_change ~ slope_before + release_dummy +
         onset_dummy + mean_post_vel + mean_post_action -- is the adjusted
         model reported in the paper; its point estimate is exact but its
         printed p-value is a naive (non-clustered) OLS t-test. The
         cluster-valid uncertainty for this same coefficient is computed
         separately in bootstrap_confound_adjusted_effect.py.
Part D: onset severity split (cheap diagnostic using already-logged object
         velocity; not used in the paper's reported numbers).

OLS is implemented by hand (closed-form, via numpy.linalg.lstsq + standard
OLS SE formula).
"""
import json
import os
import pickle

import numpy as np
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
REL_TIME = np.arange(-3, 7)  # for the 10 predicted frames (pixel MAE curves)


def load_all():
    with open(os.path.join(OUT_DIR, "pilot_episodes.pkl"), "rb") as f:
        episodes = pickle.load(f)
    with open(os.path.join(OUT_DIR, "rq1_pilot_events.json")) as f:
        events = json.load(f)
    curves = np.load(os.path.join(OUT_DIR, "rq1_pilot_curves.npz"))
    return episodes, events, curves


def curve_stats(curve):
    before_idx = np.where(REL_TIME < 0)[0]
    after_idx = np.where(REL_TIME >= 0)[0]
    slope_before = np.diff(curve[before_idx]).mean()
    slope_after = np.diff(curve[after_idx]).mean()
    return {
        "slope_before": slope_before,
        "slope_after": slope_after,
        "slope_change": slope_after - slope_before,
        "error_at_t0": curve[before_idx[-1] + 1],  # index of relative time 0
        "mean_pre_error": curve[before_idx].mean(),
        "mean_post_error": curve[after_idx].mean(),
    }


def window_motion_stats(log, t_event, rel_lo=-5, rel_hi=6):
    """Mean object velocity (cube) / action norm before (rel<0) and after
    (rel>=0) the event, using the raw per-timestep logs (available for the
    full 12-step window, not just the 10 predicted frames).
    """
    start = t_event + rel_lo
    end = t_event + rel_hi
    rel = np.arange(rel_lo, rel_hi + 1)
    vel = log["obj_velocity"][start:end + 1, 0]
    act = log["action_norm"][start:end + 1]
    pre = rel < 0
    post = rel >= 0
    return {
        "mean_pre_vel": float(vel[pre].mean()), "mean_post_vel": float(vel[post].mean()),
        "mean_pre_action": float(act[pre].mean()), "mean_post_action": float(act[post].mean()),
        "vel_at_t0": float(vel[np.where(rel == 0)[0][0]]),
        "action_at_t0": float(act[np.where(rel == 0)[0][0]]),
        "rel_time": rel, "obj_velocity": vel, "action_norm": act,
    }


def build_table(episodes, events, curves):
    rows = []
    for etype in ["onset", "release", "control"]:
        ev_list = events[f"{etype}_events"]
        curve_arr = curves[etype]
        assert len(ev_list) == curve_arr.shape[0], (etype, len(ev_list), curve_arr.shape)
        for (seed, t), curve in zip(ev_list, curve_arr):
            log = episodes[seed]
            row = {"event_type": etype, "seed": int(seed), "t": int(t)}
            row.update(curve_stats(curve))
            row.update({k: v for k, v in window_motion_stats(log, t).items()
                        if k not in ("rel_time", "obj_velocity", "action_norm")})
            rows.append(row)
    return rows


def ols(y, X_cols, X_names):
    """Closed-form OLS with an intercept. X_cols: list of 1D arrays (predictors,
    NOT including intercept). Returns dict of coef/se/t/p per name incl 'intercept'.
    """
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
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {
        "n": n, "r2": float(r2), "dof": int(dof),
        "coefs": {name: {"beta": float(b), "se": float(s), "t": float(tv), "p": float(p)}
                  for name, b, s, tv, p in zip(names, beta, se, tvals, pvals)},
    }


def part_a(rows):
    print("\n" + "=" * 70 + "\nPART A: ceiling / saturation hypothesis\n" + "=" * 70)
    slope_before = np.array([r["slope_before"] for r in rows])
    slope_after = np.array([r["slope_after"] for r in rows])
    error_at_t0 = np.array([r["error_at_t0"] for r in rows])
    etype = np.array([r["event_type"] for r in rows])

    pear, pear_p = stats.pearsonr(slope_before, slope_after)
    spear, spear_p = stats.spearmanr(slope_before, slope_after)
    print(f"pooled (n={len(rows)}) slope_before vs slope_after: Pearson r={pear:.3f} (p={pear_p:.4f}), "
          f"Spearman rho={spear:.3f} (p={spear_p:.4f})")

    pear2, pear2_p = stats.pearsonr(error_at_t0, slope_after)
    spear2, spear2_p = stats.spearmanr(error_at_t0, slope_after)
    print(f"pooled error_at_t0 vs slope_after: Pearson r={pear2:.3f} (p={pear2_p:.4f}), "
          f"Spearman rho={spear2:.3f} (p={spear2_p:.4f})")

    release_dummy = (etype == "release").astype(float)
    onset_dummy = (etype == "onset").astype(float)

    model1 = ols(slope_after, [slope_before, release_dummy, onset_dummy],
                 ["slope_before", "release_dummy", "onset_dummy"])
    print("\nModel 1: slope_after ~ slope_before + release_dummy + onset_dummy")
    for k, v in model1["coefs"].items():
        print(f"  {k:16s} beta={v['beta']:+.5f}  se={v['se']:.5f}  t={v['t']:+.2f}  p={v['p']:.4f}")
    print(f"  R^2={model1['r2']:.3f}")

    slope_change = slope_after - slope_before
    model2 = ols(slope_change, [slope_before, release_dummy, onset_dummy],
                 ["slope_before", "release_dummy", "onset_dummy"])
    print("\nModel 2: slope_change ~ slope_before + release_dummy + onset_dummy")
    for k, v in model2["coefs"].items():
        print(f"  {k:16s} beta={v['beta']:+.5f}  se={v['se']:.5f}  t={v['t']:+.2f}  p={v['p']:.4f}")
    print(f"  R^2={model2['r2']:.3f}")

    model3 = ols(slope_change, [slope_before, error_at_t0, release_dummy, onset_dummy],
                 ["slope_before", "error_at_t0", "release_dummy", "onset_dummy"])
    print("\nModel 3 (+error_at_t0 covariate): slope_change ~ slope_before + error_at_t0 + release_dummy + onset_dummy")
    for k, v in model3["coefs"].items():
        print(f"  {k:16s} beta={v['beta']:+.5f}  se={v['se']:.5f}  t={v['t']:+.2f}  p={v['p']:.4f}")
    print(f"  R^2={model3['r2']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"onset": "tab:red", "release": "tab:blue", "control": "tab:gray"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for t in ["control", "onset", "release"]:
        mask = etype == t
        axes[0].scatter(slope_before[mask], slope_after[mask], label=t, color=colors[t], alpha=0.75)
        axes[1].scatter(error_at_t0[mask], slope_after[mask], label=t, color=colors[t], alpha=0.75)
    axes[0].set_xlabel("slope_before"); axes[0].set_ylabel("slope_after")
    axes[0].set_title(f"slope_before vs slope_after (pooled r={pear:.2f})")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(0, color="k", lw=0.5)
    axes[0].legend()
    axes[1].set_xlabel("error_at_t0"); axes[1].set_ylabel("slope_after")
    axes[1].set_title(f"error_at_t0 vs slope_after (pooled r={pear2:.2f})")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_partA_ceiling.png"), dpi=130)
    print(f"\nSaved {OUT_DIR}/rq1_partA_ceiling.png")

    return {
        "pearson_before_after": [pear, pear_p], "spearman_before_after": [spear, spear_p],
        "pearson_t0_after": [pear2, pear2_p], "spearman_t0_after": [spear2, spear2_p],
        "model1_slope_after": model1, "model2_slope_change": model2, "model3_slope_change_with_t0": model3,
    }


def part_b(rows):
    print("\n" + "=" * 70 + "\nPART B: episode-cluster bootstrap\n" + "=" * 70)
    seeds = sorted(set(r["seed"] for r in rows))
    by_seed_type = {}
    for r in rows:
        by_seed_type.setdefault((r["seed"], r["event_type"]), []).append(r["slope_change"])

    rng = np.random.default_rng(0)
    n_boot = 5000

    def cluster_bootstrap_diff(type_a, type_b):
        diffs = np.zeros(n_boot)
        for b in range(n_boot):
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            vals_a, vals_b = [], []
            for s in sampled:
                vals_a.extend(by_seed_type.get((s, type_a), []))
                vals_b.extend(by_seed_type.get((s, type_b), []))
            if len(vals_a) == 0 or len(vals_b) == 0:
                diffs[b] = np.nan
                continue
            diffs[b] = np.mean(vals_a) - np.mean(vals_b)
        diffs = diffs[~np.isnan(diffs)]
        obs = np.mean([v for k in by_seed_type if k[1] == type_a for v in by_seed_type[k]]) - \
              np.mean([v for k in by_seed_type if k[1] == type_b for v in by_seed_type[k]])
        ci = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
        return obs, ci, p, diffs

    obs_on, ci_on, p_on, _ = cluster_bootstrap_diff("onset", "control")
    obs_rel, ci_rel, p_rel, _ = cluster_bootstrap_diff("release", "control")

    print(f"n episodes (clusters) = {len(seeds)}")
    print(f"onset - control slope_change:   obs={obs_on:+.5f}  cluster-95%CI={ci_on}  p~{p_on:.4f}")
    print(f"release - control slope_change: obs={obs_rel:+.5f}  cluster-95%CI={ci_rel}  p~{p_rel:.4f}")

    return {
        "n_episodes": len(seeds),
        "onset_vs_control": {"obs": float(obs_on), "ci95": ci_on.tolist(), "p": float(p_on)},
        "release_vs_control": {"obs": float(obs_rel), "ci95": ci_rel.tolist(), "p": float(p_rel)},
    }


def part_c(episodes, events, rows):
    print("\n" + "=" * 70 + "\nPART C: motion / action confound\n" + "=" * 70)
    rel_lo, rel_hi = -5, 6
    rel = np.arange(rel_lo, rel_hi + 1)
    aligned = {"onset": {"vel": [], "act": []}, "release": {"vel": [], "act": []}, "control": {"vel": [], "act": []}}
    for etype in ["onset", "release", "control"]:
        for seed, t in events[f"{etype}_events"]:
            log = episodes[seed]
            ws = window_motion_stats(log, t, rel_lo, rel_hi)
            aligned[etype]["vel"].append(ws["obj_velocity"])
            aligned[etype]["act"].append(ws["action_norm"])
        aligned[etype]["vel"] = np.array(aligned[etype]["vel"])
        aligned[etype]["act"] = np.array(aligned[etype]["act"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"onset": "tab:red", "release": "tab:blue", "control": "tab:gray"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for etype in ["onset", "release", "control"]:
        m_vel = aligned[etype]["vel"].mean(axis=0)
        m_act = aligned[etype]["act"].mean(axis=0)
        axes[0].plot(rel, m_vel, marker="o", label=etype, color=colors[etype])
        axes[1].plot(rel, m_act, marker="o", label=etype, color=colors[etype])
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_title("Mean cube speed aligned to event"); axes[0].set_xlabel("time rel. to event"); axes[0].set_ylabel("|velocity|")
    axes[0].legend()
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_title("Mean action norm aligned to event"); axes[1].set_xlabel("time rel. to event"); axes[1].set_ylabel("|action|")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_partC_motion.png"), dpi=130)
    print(f"Saved {OUT_DIR}/rq1_partC_motion.png")

    for etype in ["onset", "release", "control"]:
        v = aligned[etype]["vel"]
        a = aligned[etype]["act"]
        pre = rel < 0
        post = rel >= 0
        print(f"[{etype}] mean pre-vel={v[:,pre].mean():.4f} mean post-vel={v[:,post].mean():.4f}  "
              f"mean pre-action={a[:,pre].mean():.4f} mean post-action={a[:,post].mean():.4f}")

    # Parsimonious regression including motion/action covariates
    slope_after = np.array([r["slope_after"] for r in rows])
    slope_before = np.array([r["slope_before"] for r in rows])
    slope_change = slope_after - slope_before
    etype_arr = np.array([r["event_type"] for r in rows])
    release_dummy = (etype_arr == "release").astype(float)
    onset_dummy = (etype_arr == "onset").astype(float)
    mean_post_vel = np.array([r["mean_post_vel"] for r in rows])
    mean_post_action = np.array([r["mean_post_action"] for r in rows])

    model4 = ols(slope_change, [slope_before, release_dummy, onset_dummy, mean_post_vel, mean_post_action],
                 ["slope_before", "release_dummy", "onset_dummy", "mean_post_vel", "mean_post_action"])
    print("\nModel 4: slope_change ~ slope_before + release_dummy + onset_dummy + mean_post_vel + mean_post_action")
    for k, v in model4["coefs"].items():
        print(f"  {k:18s} beta={v['beta']:+.5f}  se={v['se']:.5f}  t={v['t']:+.2f}  p={v['p']:.4f}")
    print(f"  R^2={model4['r2']:.3f}")

    return {"aligned_means": {e: {"vel": aligned[e]["vel"].mean(axis=0).tolist(),
                                    "act": aligned[e]["act"].mean(axis=0).tolist()} for e in aligned},
            "model4_with_motion": model4}


def part_d(episodes, events, curves):
    print("\n" + "=" * 70 + "\nPART D: onset severity split\n" + "=" * 70)
    onset_events = events["onset_events"]
    onset_curves = curves["onset"]
    severities = []
    for seed, t in onset_events:
        log = episodes[seed]
        severities.append(log["obj_velocity"][t, 0])  # cube speed AT onset
    severities = np.array(severities)
    median = np.median(severities)
    low_mask = severities <= median
    high_mask = ~low_mask
    print(f"onset severity (cube speed at t=0): median={median:.4f}, "
          f"low n={low_mask.sum()}, high n={high_mask.sum()}")

    def stats_for(mask):
        curves_sub = onset_curves[mask]
        before_idx = np.where(REL_TIME < 0)[0]
        after_idx = np.where(REL_TIME >= 0)[0]
        slope_before = np.diff(curves_sub[:, before_idx], axis=1).mean(axis=1)
        slope_after = np.diff(curves_sub[:, after_idx], axis=1).mean(axis=1)
        return slope_before, slope_after, curves_sub.mean(axis=0)

    sb_low, sa_low, mean_low = stats_for(low_mask)
    sb_high, sa_high, mean_high = stats_for(high_mask)
    print(f"LOW-impact onsets:  slope_before={sb_low.mean():.5f} slope_after={sa_low.mean():.5f} "
          f"change={sa_low.mean()-sb_low.mean():+.5f}")
    print(f"HIGH-impact onsets: slope_before={sb_high.mean():.5f} slope_after={sa_high.mean():.5f} "
          f"change={sa_high.mean()-sb_high.mean():+.5f}")

    t_stat, p_val = stats.ttest_ind(sa_high - sb_high, sa_low - sb_low, equal_var=False)
    print(f"Welch t-test high vs low slope_change: t={t_stat:.3f}, p={p_val:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4.5))
    plt.plot(REL_TIME, mean_low, marker="o", label=f"low-impact (n={low_mask.sum()})", color="tab:orange")
    plt.plot(REL_TIME, mean_high, marker="o", label=f"high-impact (n={high_mask.sum()})", color="tab:purple")
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.xlabel("time relative to contact onset"); plt.ylabel("mean abs pixel error")
    plt.title("Onset severity split (median split by cube speed at t=0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq1_partD_severity.png"), dpi=130)
    print(f"Saved {OUT_DIR}/rq1_partD_severity.png")

    return {
        "median_severity": float(median), "n_low": int(low_mask.sum()), "n_high": int(high_mask.sum()),
        "low_slope_change": float(sa_low.mean() - sb_low.mean()),
        "high_slope_change": float(sa_high.mean() - sb_high.mean()),
        "welch_t": float(t_stat), "welch_p": float(p_val),
    }


def main():
    episodes, events, curves = load_all()
    rows = build_table(episodes, events, curves)

    result_a = part_a(rows)
    result_b = part_b(rows)
    result_c = part_c(episodes, events, rows)
    result_d = part_d(episodes, events, curves)

    with open(os.path.join(OUT_DIR, "rq1_confound_results.json"), "w") as f:
        json.dump({"part_a": result_a, "part_b": result_b, "part_c": result_c, "part_d": result_d},
                   f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"\nSaved combined results to {OUT_DIR}/rq1_confound_results.json")


if __name__ == "__main__":
    main()
