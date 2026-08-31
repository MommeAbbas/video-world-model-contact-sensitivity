"""
Final paper figures, built only from already-frozen, already-verified data
(analysis/analyze_event_centered_sensitivity.py's results,
analysis/analyze_physical_interpretability.py's results,
validation/validate_cube_tracker.py's error arrays,
analysis/select_qualitative_examples.py's frame arrays). No new experiment,
reconstruction, tracking, or analysis is performed here.

build_figure1 -> Figure 1 (intervention mechanics: fig1_panel_ctx0/ctx1/unpert/pert/diff).
build_figure2 -> Figure 2 (event-centered sensitivity profile: fig2_panel_profile/contrast).
build_figure3 -> not embedded in the current paper, but computes the Spearman
    correlation and tracker-resolution band cited in the paper's Section
    "Qualitative and Physical Interpretation" (rho=0.645, p<0.0001, n=119;
    median tracker error 1.11-1.27px), so it is kept even though its plot
    is not one of the paper's included figures.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
QUAL_DIR = os.path.join(OUT_DIR, "qualitative")
FIG_DIR = os.path.join(QUAL_DIR, "final_figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

DIFF_MAG = 6.0  # same, explicitly-stated contrast multiplier used throughout


def chw_to_hwc(chw):
    return np.clip(chw, 0, 1).transpose(1, 2, 0)


def savefig_all(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300 if ext == "png" else None,
                     bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Figure 1 panels have no baked-in title/label text; panel letters (a)-(e)
# are added in the LaTeX source via \subcaption.
def _bare_image_fig(img, figsize=(1.9, 1.9)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(chw_to_hwc(img), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def build_figure1():
    d = np.load(os.path.join(QUAL_DIR, "release_dt-3_frames.npz"))
    ctx0, ctx1 = d["context"][0], d["context"][1]
    frame_a, frame_b = d["frame_base"], d["frame_pert"]

    raw_diff = np.abs(frame_a - frame_b).mean(axis=0)  # (H,W) raw mean-abs-RGB diff, [0,1]
    enhanced = np.clip(raw_diff * DIFF_MAG, 0, 1)

    for name, img in [("fig1_panel_ctx0", ctx0), ("fig1_panel_ctx1", ctx1),
                       ("fig1_panel_unpert", frame_a), ("fig1_panel_pert", frame_b)]:
        fig = _bare_image_fig(img)
        savefig_all(fig, name)

    fig, ax = plt.subplots(figsize=(2.3, 1.9))
    im = ax.imshow(enhanced, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels([f"0", f"{1/DIFF_MAG:.2f}"])
    cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0, right=0.82, top=1, bottom=0)
    savefig_all(fig, "fig1_panel_diff")

    print("Saved Figure 1 panels (PNG + PDF): ctx0, ctx1, unpert, pert, diff")


# Figure 2's two panels are saved separately, no baked-in (a)/(b) titles
# (added via LaTeX \subcaption).
def build_figure2():
    with open(os.path.join(OUT_DIR, "rq2_event_centered_analysis_results.json")) as f:
        analysis = json.load(f)
    profiles = analysis["profiles"]
    contrasts = analysis["contrasts"]
    offsets = [-3, -2, -1, 0, 1, 2, 3]

    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    # Panel (a): absolute profiles
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for et, color in [("onset", "tab:red"), ("release", "tab:blue")]:
        dts, means, los, his = [], [], [], []
        for dt in offsets:
            p = profiles[et][str(dt)] if str(dt) in profiles[et] else profiles[et][dt]
            if p["n_events"] > 0:
                dts.append(dt); means.append(p["mean"]); los.append(p["ci_lo"]); his.append(p["ci_hi"])
        ax.plot(dts, means, marker="o", ms=5, label=et, color=color, lw=1.8)
        ax.fill_between(dts, los, his, alpha=0.18, color=color, linewidth=0)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"mean downstream divergence $\bar{D}$")
    ax.set_xticks(offsets)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    savefig_all(fig, "fig2_panel_profile")

    # Panel (b): within-event contrasts
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for et, color in [("onset", "tab:red"), ("release", "tab:blue")]:
        dts, means, los, his = [], [], [], []
        for dt in offsets:
            if dt == 0:
                continue
            c = contrasts[et][str(dt)] if str(dt) in contrasts[et] else contrasts[et][dt]
            if c["n_events"] > 0:
                dts.append(dt); means.append(c["mean"]); los.append(c["ci_lo"]); his.append(c["ci_hi"])
        ax.plot(dts, means, marker="o", ms=5, label=et, color=color, lw=1.8)
        ax.fill_between(dts, los, his, alpha=0.18, color=color, linewidth=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\Delta t$"); ax.set_ylabel(r"$C(\Delta t) = \bar{D}(\Delta t) - \bar{D}(0)$")
    ax.set_xticks(offsets)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    savefig_all(fig, "fig2_panel_contrast")

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })
    print("Saved Figure 2 panels (PNG + PDF): profile, contrast")


# Not embedded in the current paper; kept because it reproduces the
# Spearman/tracker-band numbers (see module docstring).
def build_figure3():
    with open(os.path.join(QUAL_DIR, "physical_interpretability_results.json")) as f:
        rows = json.load(f)
    usable = [r for r in rows if not r["flagged"] and r["mean_centroid_sep_future"] is not None]

    div = np.array([r["mean_downstream_divergence"] for r in usable])
    sep = np.array([r["mean_centroid_sep_future"] for r in usable])

    gt_err = np.load(os.path.join(QUAL_DIR, "tracker_validation_errors.npz"))["errors"]
    vq_err = np.load(os.path.join(QUAL_DIR, "tracker_validation_vqrecon_errors.npy"))
    band_lo, band_hi = float(np.median(gt_err)), float(np.median(vq_err))

    from scipy import stats
    rho, p = stats.spearmanr(div, sep)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.2, 1]})

    ax = axes[0]
    ax.axhspan(min(band_lo, band_hi), max(band_lo, band_hi), color="gray", alpha=0.25,
                label=f"tracker resolution\n(median GT error {band_lo:.2f}-{band_hi:.2f}px)")
    ax.scatter(div, sep, s=18, alpha=0.35, color="tab:blue", edgecolor="none")

    order = np.argsort(div)
    n_bins = 8
    bin_edges = np.quantile(div, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6
    bin_centers, bin_medians = [], []
    for i in range(n_bins):
        mask = (div >= bin_edges[i]) & (div < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(div[mask].mean())
            bin_medians.append(np.median(sep[mask]))
    ax.plot(bin_centers, bin_medians, color="tab:red", marker="D", ms=5, lw=1.5,
             label="binned median trend")

    ax.set_xlabel("mean downstream token divergence")
    ax.set_ylabel("A-vs-B tracked cube centroid separation (px)")
    ax.set_title(f"(a) Token divergence vs. tracked physical displacement\n"
                  f"Spearman $\\rho$={rho:.3f} (p<0.0001), n={len(usable)}")
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    # Panel (b): four frozen profile categories, aggregate distributions only
    ax2 = axes[1]
    cats = [("onset", -3, "onset\n$\\Delta t{=}{-3}$"), ("onset", 2, "onset\n$\\Delta t{=}{+2}$"),
            ("release", -3, "release\n$\\Delta t{=}{-3}$"), ("release", 2, "release\n$\\Delta t{=}{+2}$")]
    positions, data = [], []
    for i, (et, dt, lab) in enumerate(cats):
        vals = [r["mean_centroid_sep_future"] for r in usable if r["event_type"] == et and r["dt"] == dt]
        data.append(vals)
        positions.append(i)
    bp = ax2.boxplot(data, positions=positions, widths=0.5, showmeans=True, meanline=True,
                       patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue"); patch.set_alpha(0.3)
    ax2.axhspan(min(band_lo, band_hi), max(band_lo, band_hi), color="gray", alpha=0.25, zorder=0)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([c[2] for c in cats], fontsize=6.5)
    ax2.set_ylabel("centroid separation (px)")
    ax2.set_title("(b) Frozen profile categories\n(aggregate only)")

    plt.tight_layout()
    savefig_all(fig, "figure3_physical_interpretability")
    print("Saved Figure 3 (PNG + PDF)")
    return rho, len(usable), band_lo, band_hi


if __name__ == "__main__":
    build_figure1()
    build_figure2()
    rho, n, lo, hi = build_figure3()
    print(f"\nFigure 3 stats used: Spearman rho={rho:.4f}, n={n}, tracker band=[{lo:.2f},{hi:.2f}]px")
