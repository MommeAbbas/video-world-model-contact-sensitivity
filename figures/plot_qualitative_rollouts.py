"""
Qualitative rollout figures (paper Section "Qualitative and Physical
Interpretation", Figure 3): large decoded A/B frames with a localized
difference overlay, built only from the already objectively-selected,
already exactly-reconstructed frame arrays saved by
analysis/select_qualitative_examples.py. No new example selection, no new
model inference, no metric changes, no new tracker/statistical analysis.

One frozen difference-overlay rule, applied identically to every panel in
both figures (never rescaled per-panel): raw mean-abs-RGB difference between
the A and B decoded frame, thresholded at DIFF_THRESHOLD = 0.05 (5% of the
[0,1] pixel scale), overlaid as a semi-transparent red region on top of the A
(unperturbed) decoded frame. This threshold was chosen by inspecting the
pooled raw-diff distribution across all already-existing frame-pairs before
building any figure; it is not tuned per example or per figure.

figureQ1_release_rollouts is the paper's embedded Figure 3. figureQ2_onset
_rollouts is built with the identical frozen rule to substantiate the
paper's text ("Figure~\\ref{fig:qualitative} shows release; onset looks the
same") even though it is not itself included in the paper.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUAL_DIR = os.path.join(REPO_ROOT, "outputs", "qualitative")
FIG_DIR = os.path.join(QUAL_DIR, "final_figures")
os.makedirs(FIG_DIR, exist_ok=True)

DIFF_THRESHOLD = 0.05  # frozen, stated once, identical for every panel in both figures

plt.rcParams.update({"font.size": 11})


def chw_to_hwc(chw):
    return np.clip(chw, 0, 1).transpose(1, 2, 0)


def overlay_diff(a_chw, b_chw, threshold=DIFF_THRESHOLD, color=(1.0, 0.0, 0.0), alpha=0.55):
    """A (unperturbed) frame with a semi-transparent color region wherever
    the raw mean-abs-RGB diff to B exceeds the frozen threshold."""
    base = chw_to_hwc(a_chw).copy()
    diff = np.abs(a_chw - b_chw).mean(axis=0)
    mask = diff > threshold
    overlay = base.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask, (1 - alpha) * base[..., c] + alpha * color[c], base[..., c])
    return overlay, mask


def build_condition_block(fig, gs, col0, key):
    """One condition occupies 4 columns (timesteps) x 3 rows (A, B, diff) at
    grid column offset col0. No condition-name title is drawn here -- which
    condition is which is stated in the LaTeX caption instead. Row labels
    (A/B/diff) and column labels ($t_i$, ...) are kept as short,
    axis-label-equivalent text; the per-panel %-pixel numbers are not drawn
    (the qualitative text already reports the representative percentages)."""
    d = np.load(os.path.join(QUAL_DIR, f"{key}_frames.npz"))
    a_frames = [d["frame_base"]] + [d["recon_A_future"][i] for i in range(3)]
    b_frames = [d["frame_pert"]] + [d["recon_B_future"][i] for i in range(3)]
    col_titles = ["$t_i$", "$t_i{+}1$", "$t_i{+}2$", "$t_i{+}3$"]
    is_left = col0 == 0

    for c in range(4):
        ax_a = fig.add_subplot(gs[0, col0 + c])
        ax_a.imshow(chw_to_hwc(a_frames[c]), interpolation="nearest")
        ax_a.set_xticks([]); ax_a.set_yticks([])
        ax_a.set_title(col_titles[c], fontsize=10)
        if is_left and c == 0:
            ax_a.set_ylabel("A", fontsize=10, rotation=0, labelpad=10, va="center")

        ax_b = fig.add_subplot(gs[1, col0 + c])
        ax_b.imshow(chw_to_hwc(b_frames[c]), interpolation="nearest")
        ax_b.set_xticks([]); ax_b.set_yticks([])
        if is_left and c == 0:
            ax_b.set_ylabel("B", fontsize=10, rotation=0, labelpad=10, va="center")

        overlay, mask = overlay_diff(a_frames[c], b_frames[c])
        ax_d = fig.add_subplot(gs[2, col0 + c])
        ax_d.imshow(overlay, interpolation="nearest")
        ax_d.set_xticks([]); ax_d.set_yticks([])
        if is_left and c == 0:
            ax_d.set_ylabel("diff", fontsize=10, rotation=0, labelpad=10, va="center")


def build_figure(conditions, out_name):
    """conditions: list of (npz_key, label) -- exactly 2, placed side by side
    (horizontal layout) rather than stacked, to keep the figure wide and
    short. `label` is not drawn in the image; it is returned for use in the
    LaTeX caption."""
    fig = plt.figure(figsize=(9.5, 3.4))
    gs = GridSpec(3, 8, figure=fig, height_ratios=[1, 1, 0.62], hspace=0.08,
                   wspace=0.06)
    for i, (key, _label) in enumerate(conditions):
        build_condition_block(fig, gs, i * 4, key)

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{out_name}.{ext}"),
                     dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {out_name} (PNG + PDF)")


if __name__ == "__main__":
    build_figure(
        [("release_dt-3", "Release, $\\Delta t=-3$ (higher-sensitivity)"),
         ("release_dt+2", "Release, $\\Delta t=+2$ (lower-sensitivity)")],
        "figureQ1_release_rollouts")
    build_figure(
        [("onset_dt-3", "Onset, $\\Delta t=-3$ (lower-sensitivity)"),
         ("onset_dt+2", "Onset, $\\Delta t=+2$ (higher-sensitivity)")],
        "figureQ2_onset_rollouts")
