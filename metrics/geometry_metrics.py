"""Geometry-oriented defect measurement metrics for the revision protocol.

All functions operate on binary masks at the original image size.  The
implementation follows the public revision protocol and keeps empty-mask
handling explicit so that complete misses do not produce artificial RAE values
through tiny denominators.
"""

import math
from collections import deque

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - optional dependency
    ndimage = None

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except Exception:  # pragma: no cover - optional dependency
    skimage_skeletonize = None


EPS = 1e-6


def as_binary(mask):
    return (np.asarray(mask) > 0).astype(np.uint8)


def image_diagonal(mask):
    h, w = np.asarray(mask).shape[:2]
    return float(math.sqrt(h * h + w * w))


def area(mask):
    return float(as_binary(mask).sum())


def erode3x3(mask):
    mask = as_binary(mask)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if cv2 is not None:
        return cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return eroded.astype(np.uint8)


def boundary(mask):
    """Return a one-pixel binary boundary as mask - 3x3_erode(mask)."""
    mask = as_binary(mask)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return ((mask - erode3x3(mask)) > 0).astype(np.uint8)


def perimeter(mask):
    """Boundary pixel count used in B = P^2 / (4*pi*A + eps)."""
    return float(boundary(mask).sum())


def boundary_irregularity(mask):
    mask = as_binary(mask)
    a = area(mask)
    if a <= 0:
        return 0.0
    p = perimeter(mask)
    return float((p * p) / (4.0 * math.pi * a + EPS))


def skeletonize_mask(mask):
    mask = as_binary(mask).astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.uint8)
    if skimage_skeletonize is not None:
        return skimage_skeletonize(mask).astype(np.uint8)
    if cv2 is not None and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0).astype(np.uint8)
    # Fallback keeps the binary mask so downstream code remains deterministic.
    return mask.astype(np.uint8)


def skeleton_length(mask_or_skeleton, already_skeleton=False):
    """Return 8-neighbor graph length of a single-pixel skeleton.

    Horizontal and vertical undirected edges contribute 1.0.  Diagonal
    undirected edges contribute sqrt(2).  Each edge is counted once.
    """
    skel = as_binary(mask_or_skeleton) if already_skeleton else skeletonize_mask(mask_or_skeleton)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return 0.0
    skel_bool = skel.astype(bool)
    length = 0.0
    h, w = skel.shape
    for y, x in zip(ys, xs):
        if x + 1 < w and skel_bool[y, x + 1]:
            length += 1.0
        if y + 1 < h and skel_bool[y + 1, x]:
            length += 1.0
        if y + 1 < h and x + 1 < w and skel_bool[y + 1, x + 1]:
            length += math.sqrt(2.0)
        if y + 1 < h and x - 1 >= 0 and skel_bool[y + 1, x - 1]:
            length += math.sqrt(2.0)
    return float(length)


def component_count(mask, min_area=1):
    """Count connected components with 8-neighbor connectivity."""
    mask = as_binary(mask)
    if mask.max() == 0:
        return 0
    if cv2 is not None:
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return int(sum(1 for idx in range(1, num) if stats[idx, cv2.CC_STAT_AREA] >= int(min_area)))
    if ndimage is not None:
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        if min_area <= 1:
            return int(count)
        return int(sum(1 for idx in range(1, count + 1) if np.sum(labels == idx) >= int(min_area)))
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        pixels = 0
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            pixels += 1
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if pixels >= int(min_area):
            count += 1
    return int(count)


def centroid(mask):
    ys, xs = np.nonzero(as_binary(mask))
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def centroid_error(pred, gt):
    pred = as_binary(pred)
    gt = as_binary(gt)
    if gt.sum() == 0:
        return None
    c_gt = centroid(gt)
    c_pred = centroid(pred)
    if c_gt is not None and c_pred is None:
        return image_diagonal(gt)
    if c_gt is not None and c_pred is not None:
        return float(np.linalg.norm(c_pred - c_gt))
    return None


def boundary_distance_error(pred, gt):
    pred = as_binary(pred)
    gt = as_binary(gt)
    diag = image_diagonal(gt)
    if gt.sum() == 0 and pred.sum() == 0:
        return None
    if gt.sum() > 0 and pred.sum() == 0:
        return diag
    if gt.sum() == 0 and pred.sum() > 0:
        return None
    pred_b = boundary(pred)
    gt_b = boundary(gt)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return diag
    if cv2 is not None:
        dist_to_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, 3)
        return float(0.5 * dist_to_gt[pred_b > 0].mean() + 0.5 * dist_to_pred[gt_b > 0].mean())
    if ndimage is not None:
        dist_to_gt = ndimage.distance_transform_edt(1 - gt_b)
        dist_to_pred = ndimage.distance_transform_edt(1 - pred_b)
        return float(0.5 * dist_to_gt[pred_b > 0].mean() + 0.5 * dist_to_pred[gt_b > 0].mean())
    pred_pts = np.column_stack(np.nonzero(pred_b))
    gt_pts = np.column_stack(np.nonzero(gt_b))
    dists = ((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=2) ** 0.5
    return float(0.5 * dists.min(axis=1).mean() + 0.5 * dists.min(axis=0).mean())


def sample_geometry_record(pred, gt, min_component_area=1):
    """Per-sample geometry record.

    RAE, LE, BDE, CE, and ConnE are defined for GT-positive samples.  GT-empty
    rows retain diagnostic areas/counts but use None for geometry errors.
    """
    pred = as_binary(pred)
    gt = as_binary(gt)
    gt_area = area(gt)
    pred_area = area(pred)
    gt_len = skeleton_length(gt)
    pred_len = skeleton_length(pred)
    gt_count = component_count(gt, min_area=min_component_area)
    pred_count = component_count(pred, min_area=min_component_area)
    record = {
        "gt_area": gt_area,
        "pred_area": pred_area,
        "gt_length": gt_len,
        "pred_length": pred_len,
        "gt_irregularity": boundary_irregularity(gt),
        "pred_irregularity": boundary_irregularity(pred),
        "gt_component_count": gt_count,
        "pred_component_count": pred_count,
        "prediction_empty": bool(pred_area <= 0),
    }
    if gt_area <= 0:
        record.update({"rae": None, "le": None, "bde": None, "ce": None, "conne": None})
        return record
    record["rae"] = abs(pred_area - gt_area) / max(gt_area, EPS)
    record["le"] = abs(pred_len - gt_len) / max(gt_len, EPS)
    record["bde"] = boundary_distance_error(pred, gt)
    record["ce"] = centroid_error(pred, gt)
    record["conne"] = abs(pred_count - gt_count)
    return record


def mean_valid(records, key):
    values = [r.get(key) for r in records if r.get(key) is not None]
    return "-" if not values else float(np.mean(values))


def aggregate_geometry_records(records):
    records = list(records)
    defect_records = [r for r in records if r.get("gt_area", 0) > 0]
    out = {
        "total_samples": len(records),
        "GT_positive_samples": len(defect_records),
        "GT_empty_samples": len(records) - len(defect_records),
        "prediction_empty_samples": sum(1 for r in records if r.get("prediction_empty")),
        "both_empty_samples": sum(1 for r in records if r.get("gt_area", 0) <= 0 and r.get("prediction_empty")),
    }
    if not defect_records:
        for key in ["RAE", "LE", "BDE", "CE", "ConnE"]:
            out[key] = "-"
        return out
    out["RAE"] = mean_valid(defect_records, "rae")
    out["LE"] = mean_valid(defect_records, "le")
    out["BDE"] = mean_valid(defect_records, "bde")
    out["CE"] = mean_valid(defect_records, "ce")
    out["ConnE"] = mean_valid(defect_records, "conne")
    return out
