"""
Deterministic, non-learned cube tracker for the physical-interpretability
audit of the already-frozen event-centered experiment. Color-segmentation +
connected-components only -- no learned detector, nothing trained.

Cube (object index 0) is rgba=[0.8,0.1,0.1,1] (red); the other three objects
are blue [0.1,0.1,0.8], green [0.1,0.8,0.1], yellow [0.8,0.8,0.1]
(push_center_multi_lite.py). "Red minus max(green,blue)" isolates red from
blue/green (low red) and from yellow (high green) without needing a learned
model.

MUST be validated (validation/validate_cube_tracker.py) against simulator-projected
ground truth before being trusted on any generated/decoded frame.
"""
import numpy as np
from scipy import ndimage


def cube_mask(frame_hwc, red_margin=0.12, min_area=3):
    """frame_hwc: (H,W,3) float in [0,1]. Returns boolean mask of the
    largest connected "red" component, or None if no component found.
    """
    r, g, b = frame_hwc[..., 0], frame_hwc[..., 1], frame_hwc[..., 2]
    redness = r - np.maximum(g, b)
    mask = redness > red_margin
    if not mask.any():
        return None
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best = np.argmax(sizes) + 1
    if sizes[best - 1] < min_area:
        return None
    return labeled == best


def cube_centroid(frame_chw_or_hwc):
    """Accepts (3,H,W) or (H,W,3) float [0,1]. Returns (row,col) centroid in
    pixel coordinates, or None if the cube could not be located.
    """
    arr = frame_chw_or_hwc
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = arr.transpose(1, 2, 0)
    mask = cube_mask(arr)
    if mask is None:
        return None
    rows, cols = np.where(mask)
    return float(rows.mean()), float(cols.mean())
