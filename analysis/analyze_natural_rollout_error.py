"""
Descriptive summary and plots for the natural rollout-error experiment (paper
Section "Contact-Conditioned Rollout Dynamics"): mean +/- SEM curves, error
increment (delta e_t), and a segmented slope-before/after summary
(relative time < 0 vs. >= 0) for onset/release/control.

This computes only descriptive statistics and plots -- the bootstrap
comparison against control reported in the paper is a separate script,
bootstrap_natural_rollout_effect.py, since it requires the per-event (not
pooled-mean) slope values.
"""
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
REL_TIME = np.arange(-3, 7)  # relative to event, for the 10 predicted frames


def mean_sem(curves):
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0]) if curves.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def slope(curves, rel_time, before=True):
    mask = rel_time < 0 if before else rel_time >= 0
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return np.nan
    # simple average of consecutive differences within the segment
    diffs = np.diff(curves[:, idx], axis=1)
    return diffs.mean()


def main():
    data = np.load(os.path.join(OUT_DIR, "rq1_pilot_curves.npz"))
    onset, release, control = data["onset"], data["release"], data["control"]
    print(f"onset events: {onset.shape[0]}, release events: {release.shape[0]}, control windows: {control.shape[0]}")

    summary = {}
    for name, curves in [("onset", onset), ("release", release), ("control", control)]:
        mean, sem = mean_sem(curves)
        slope_before = slope(curves, REL_TIME, before=True)
        slope_after = slope(curves, REL_TIME, before=False)
        summary[name] = {
            "n": int(curves.shape[0]),
            "mean_error": mean.tolist(), "sem_error": sem.tolist(),
            "mean_delta": np.diff(mean).tolist(),
            "slope_before": float(slope_before), "slope_after": float(slope_after),
            "slope_change": float(slope_after - slope_before),
        }
        print(f"\n[{name}] n={curves.shape[0]}")
        print(f"  mean error per relative-time step: {np.round(mean,4).tolist()}")
        print(f"  slope before={slope_before:.5f}  slope after={slope_after:.5f}  "
              f"change={slope_after-slope_before:+.5f}")

    with open(os.path.join(OUT_DIR, "rq1_pilot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def plot_curve(ax, rel_time, mean, sem, label, color):
        ax.plot(rel_time, mean, marker="o", label=label, color=color)
        ax.fill_between(rel_time, mean - sem, mean + sem, alpha=0.2, color=color)

    m_on, s_on = mean_sem(onset)
    m_ctrl, s_ctrl = mean_sem(control)
    plot_curve(axes[0, 0], REL_TIME, m_on, s_on, f"onset (n={onset.shape[0]})", "tab:red")
    plot_curve(axes[0, 0], REL_TIME, m_ctrl, s_ctrl, f"control (n={control.shape[0]})", "tab:gray")
    axes[0, 0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Contact ONSET aligned rollout error")
    axes[0, 0].set_xlabel("time relative to contact onset (steps)")
    axes[0, 0].set_ylabel("mean abs pixel error")
    axes[0, 0].legend()

    m_rel, s_rel = mean_sem(release)
    plot_curve(axes[0, 1], REL_TIME, m_rel, s_rel, f"release (n={release.shape[0]})", "tab:blue")
    plot_curve(axes[0, 1], REL_TIME, m_ctrl, s_ctrl, f"control (n={control.shape[0]})", "tab:gray")
    axes[0, 1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Contact RELEASE aligned rollout error")
    axes[0, 1].set_xlabel("time relative to contact release (steps)")
    axes[0, 1].set_ylabel("mean abs pixel error")
    axes[0, 1].legend()

    d_on = np.diff(m_on)
    d_ctrl = np.diff(m_ctrl)
    d_rel = np.diff(m_rel)
    rel_mid = REL_TIME[:-1] + 0.5
    axes[1, 0].plot(rel_mid, d_on, marker="o", label="onset", color="tab:red")
    axes[1, 0].plot(rel_mid, d_ctrl, marker="o", label="control", color="tab:gray")
    axes[1, 0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0, color="k", linewidth=0.5)
    axes[1, 0].set_title("Error increment delta_e_t (onset vs control)")
    axes[1, 0].set_xlabel("time relative to contact onset (steps)")
    axes[1, 0].set_ylabel("delta e_t")
    axes[1, 0].legend()

    axes[1, 1].plot(rel_mid, d_rel, marker="o", label="release", color="tab:blue")
    axes[1, 1].plot(rel_mid, d_ctrl, marker="o", label="control", color="tab:gray")
    axes[1, 1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0, color="k", linewidth=0.5)
    axes[1, 1].set_title("Error increment delta_e_t (release vs control)")
    axes[1, 1].set_xlabel("time relative to contact release (steps)")
    axes[1, 1].set_ylabel("delta e_t")
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "rq1_pilot_plot.png")
    plt.savefig(plot_path, dpi=130)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()
