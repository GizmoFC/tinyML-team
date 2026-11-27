"""Click simulation for interactive-segmentation evaluation (proposal Sec. 3.3).

- First click: positive, at the point inside the target farthest from its boundary.
- Corrective clicks: at the deepest point of the largest remaining error region.
  Missed foreground (false negative) -> positive click; falsely included
  background (false positive) -> negative click.
"""

from __future__ import annotations

import cv2
import numpy as np


def _deepest_point(region: np.ndarray) -> tuple[int, int]:
    """(x, y) of the pixel in `region` farthest from its boundary.

    The region is zero-padded by 1 px so the image border also counts as a boundary.
    Ties are broken by the first index in row-major order (deterministic).
    """
    padded = np.pad(region.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def first_click(gt: np.ndarray) -> tuple[int, int, int]:
    """Return (x, y, label=1) for the initial positive click."""
    x, y = _deepest_point(gt.astype(bool))
    return x, y, 1


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx, int(stats[idx, cv2.CC_STAT_AREA])


def corrective_click(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int] | None:
    """Return (x, y, label) for the next corrective click, or None if there are no errors."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    fn_region, fn_area = _largest_component(gt & ~pred)
    fp_region, fp_area = _largest_component(pred & ~gt)
    if fn_area == 0 and fp_area == 0:
        return None
    if fn_area >= fp_area:
        x, y = _deepest_point(fn_region)
        return x, y, 1
    x, y = _deepest_point(fp_region)
    return x, y, 0


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)
