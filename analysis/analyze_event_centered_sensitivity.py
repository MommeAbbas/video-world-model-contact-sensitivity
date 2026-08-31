"""
Analysis for the event-centered temporal sensitivity profile (paper Section
"Temporal Structure of Predictive Sensitivity", Figure 2). Builds
S_onset(Delta_t)/S_release(Delta_t), within-event contrasts
C_onset/C_release, and the injection-Hamming diagnostic, all with
episode-cluster bootstrap uncertainty (cluster = episode seed).

The three PNGs saved here are quick diagnostic plots, not the paper's
Figure 2 -- see figures/plot_paper_figures.py for the final panel figure.
"""
import json
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
N_BOOT = 3000


def load():
    with open(os.path.join(OUT_DIR, "rq2_event_centered_results.json")) as f:
        return json.load(f)


def cluster_bootstrap_mean(values, seeds, n_boot=N_BOOT, seed=0):
    """Episode-cluster bootstrap CI on the mean of `values`, clustering by
    `seeds` (one seed per value, can repeat)."""
    values = np.asarray(values)
    seeds = np.asarray(seeds)
    clusters = sorted(set(seeds.tolist()))
    if len(clusters) < 2:
        return np.nan, np.nan, len(clusters)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        vals = []
        for c in sampled:
            vals.extend(values[seeds == c].tolist())
        if vals:
            means.append(np.mean(vals))
    means = np.array(means)
    ci = np.percentile(means, [2.5, 97.5])
    return ci[0], ci[1], len(clusters)


def build_profile(rows, event_type):
    """Returns dict: dt -> {mean, ci_lo, ci_hi, n_events, n_episodes}"""
    profile = {}
    for dt in OFFSETS:
        sub = [r for r in rows if r["event_type"] == event_type and r["dt"] == dt]
        if not sub:
            profile[dt] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_events": 0, "n_episodes": 0}
            continue
        vals = [r["mean_downstream_divergence"] for r in sub]
        seeds = [r["seed"] for r in sub]
        ci_lo, ci_hi, n_ep = cluster_bootstrap_mean(vals, seeds)
        profile[dt] = {"mean": float(np.mean(vals)), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                        "n_events": len(sub), "n_episodes": n_ep}
    return profile


def build_contrasts(rows, event_type):
    """C_i(dt) = D_bar_i(dt) - D_bar_i(0), for events with BOTH dt=0 and dt clean."""
    # index by (batch, seed, tau)
    by_event = {}
    for r in rows:
        if r["event_type"] != event_type:
            continue
        key = (r["batch"], r["seed"], r["tau"])
        by_event.setdefault(key, {})[r["dt"]] = r["mean_downstream_divergence"]

    contrasts = {}
    for dt in OFFSETS:
        if dt == 0:
            continue
        vals, seeds = [], []
        for (batch, seed, tau), d in by_event.items():
            if 0 in d and dt in d:
                vals.append(d[dt] - d[0])
                seeds.append(seed)
        if not vals:
            contrasts[dt] = {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_events": 0, "n_episodes": 0}
            continue
        ci_lo, ci_hi, n_ep = cluster_bootstrap_mean(vals, seeds)
        contrasts[dt] = {"mean": float(np.mean(vals)), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                          "n_events": len(vals), "n_episodes": n_ep}
    return contrasts


def hamming_diagnostic(rows, event_type):
    diag = {}
    for dt in OFFSETS:
        sub = [r["injection_hamming"] for r in rows if r["event_type"] == event_type and r["dt"] == dt]
        diag[dt] = {"mean": float(np.mean(sub)) if sub else np.nan, "n": len(sub)}
    return diag


def main():
    rows = load()
    print(f"Loaded {len(rows)} event-centered intervention records")

    profiles = {et: build_profile(rows, et) for et in ["onset", "release"]}
    contrasts = {et: build_contrasts(rows, et) for et in ["onset", "release"]}
    hammings = {et: hamming_diagnostic(rows, et) for et in ["onset", "release"]}

    for et in ["onset", "release"]:
        print(f"\n=== S_{et}(Delta_t) ===")
        for dt in OFFSETS:
            p = profiles[et][dt]
            print(f"  dt={dt:+d}: mean={p['mean']:.3f}  CI=[{p['ci_lo']:.3f},{p['ci_hi']:.3f}]  "
                  f"n_events={p['n_events']} n_episodes={p['n_episodes']}")

    for et in ["onset", "release"]:
        print(f"\n=== C_{et}(Delta_t) = D(dt) - D(0), within-event ===")
        for dt in OFFSETS:
            if dt == 0:
                continue
            c = contrasts[et][dt]
            print(f"  dt={dt:+d}: mean={c['mean']:.3f}  CI=[{c['ci_lo']:.3f},{c['ci_hi']:.3f}]  "
                  f"n_events={c['n_events']} n_episodes={c['n_episodes']}")

    for et in ["onset", "release"]:
        print(f"\n=== injection_hamming vs Delta_t ({et}) ===")
        for dt in OFFSETS:
            h = hammings[et][dt]
            print(f"  dt={dt:+d}: mean_hamming={h['mean']:.2f} (n={h['n']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: S_onset / S_release
    fig, ax = plt.subplots(figsize=(7, 5))
    for et, color in [("onset", "tab:red"), ("release", "tab:blue")]:
        dts = [dt for dt in OFFSETS if profiles[et][dt]["n_events"] > 0]
        means = [profiles[et][dt]["mean"] for dt in dts]
        lo = [profiles[et][dt]["ci_lo"] for dt in dts]
        hi = [profiles[et][dt]["ci_hi"] for dt in dts]
        ax.plot(dts, means, marker="o", label=et, color=color)
        ax.fill_between(dts, lo, hi, alpha=0.2, color=color)
        for dt in dts:
            ax.annotate(f"n={profiles[et][dt]['n_events']}", (dt, profiles[et][dt]["mean"]),
                         textcoords="offset points", xytext=(0, 8), fontsize=7, color=color)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Delta_t (t_i = tau + Delta_t)")
    ax.set_ylabel("mean downstream divergence D_bar")
    ax.set_title("Event-centered sensitivity profile S(Delta_t)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_event_centered_fig1_profiles.png"), dpi=130)

    # Figure 2: within-event contrasts
    fig, ax = plt.subplots(figsize=(7, 5))
    for et, color in [("onset", "tab:red"), ("release", "tab:blue")]:
        dts = [dt for dt in OFFSETS if dt != 0 and contrasts[et][dt]["n_events"] > 0]
        means = [contrasts[et][dt]["mean"] for dt in dts]
        lo = [contrasts[et][dt]["ci_lo"] for dt in dts]
        hi = [contrasts[et][dt]["ci_hi"] for dt in dts]
        ax.plot(dts, means, marker="o", label=et, color=color)
        ax.fill_between(dts, lo, hi, alpha=0.2, color=color)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Delta_t"); ax.set_ylabel("C(Delta_t) = D_bar(Delta_t) - D_bar(0)")
    ax.set_title("Within-event contrast profile")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_event_centered_fig2_contrasts.png"), dpi=130)

    # Figure 3: injection hamming diagnostic
    fig, ax = plt.subplots(figsize=(7, 5))
    for et, color in [("onset", "tab:red"), ("release", "tab:blue")]:
        dts = [dt for dt in OFFSETS if hammings[et][dt]["n"] > 0]
        means = [hammings[et][dt]["mean"] for dt in dts]
        ax.plot(dts, means, marker="o", label=et, color=color)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Delta_t"); ax.set_ylabel("mean injection_hamming")
    ax.set_title("Injection-Hamming calibration diagnostic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rq2_event_centered_fig3_hamming.png"), dpi=130)

    with open(os.path.join(OUT_DIR, "rq2_event_centered_analysis_results.json"), "w") as f:
        json.dump({"profiles": profiles, "contrasts": contrasts, "hamming_diagnostic": hammings},
                   f, indent=2)
    print("\nSaved figures and analysis results.")


if __name__ == "__main__":
    main()
