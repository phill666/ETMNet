from pathlib import Path

import numpy as np
from PIL import Image


try:
    from scipy import ndimage
    from scipy.stats import spearmanr as scipy_spearmanr
except Exception:
    ndimage = None
    scipy_spearmanr = None


MASK_SUFFIXES = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_binary_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def area(mask):
    return float(np.asarray(mask, dtype=bool).sum())


def bbox_length(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return max(width, height)


def component_count(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    if ndimage is not None:
        _, count = ndimage.label(mask)
        return int(count)
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    if ndimage is not None:
        eroded = ndimage.binary_erosion(mask)
    else:
        padded = np.pad(mask, 1, mode="constant")
        eroded = np.ones_like(mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                eroded &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return mask ^ eroded


def boundary_distance_error(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.hypot(*gt.shape))

    pred_boundary = _boundary(pred)
    gt_boundary = _boundary(gt)
    if ndimage is not None:
        dist_to_gt = ndimage.distance_transform_edt(~gt_boundary)
        dist_to_pred = ndimage.distance_transform_edt(~pred_boundary)
        pred_to_gt = dist_to_gt[pred_boundary].mean() if pred_boundary.any() else 0.0
        gt_to_pred = dist_to_pred[gt_boundary].mean() if gt_boundary.any() else 0.0
        return float((pred_to_gt + gt_to_pred) / 2.0)

    pred_points = np.column_stack(np.nonzero(pred_boundary))
    gt_points = np.column_stack(np.nonzero(gt_boundary))
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float(np.hypot(*gt.shape))
    dists = ((pred_points[:, None, :] - gt_points[None, :, :]) ** 2).sum(axis=2) ** 0.5
    return float((dists.min(axis=1).mean() + dists.min(axis=0).mean()) / 2.0)


def severity_label(mask, thresholds=(0.005, 0.02)):
    ratio = area(mask) / float(mask.shape[0] * mask.shape[1])
    if ratio <= thresholds[0]:
        return 0
    if ratio <= thresholds[1]:
        return 1
    return 2


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_rho(pred_values, gt_values):
    pred_values = np.asarray(pred_values, dtype=np.float64)
    gt_values = np.asarray(gt_values, dtype=np.float64)
    if len(pred_values) < 2 or np.std(pred_values) == 0 or np.std(gt_values) == 0:
        return 0.0
    if scipy_spearmanr is not None:
        value = scipy_spearmanr(pred_values, gt_values).correlation
        return 0.0 if np.isnan(value) else float(value)
    pred_rank = _rankdata(pred_values)
    gt_rank = _rankdata(gt_values)
    corr = np.corrcoef(pred_rank, gt_rank)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def sample_measurement_metrics(pred, gt, severity_thresholds=(0.005, 0.02)):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    pred_area = area(pred)
    gt_area = area(gt)
    pred_len = bbox_length(pred)
    gt_len = bbox_length(gt)
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    pred_sev = severity_label(pred, severity_thresholds)
    gt_sev = severity_label(gt, severity_thresholds)
    return {
        "RAE": abs(pred_area - gt_area) / max(gt_area, 1.0),
        "LE": abs(pred_len - gt_len),
        "BDE": boundary_distance_error(pred, gt),
        "CE": abs(pred_count - gt_count),
        "ConnE": abs(pred_count - gt_count) / max(gt_count, 1),
        "Severity_Acc": 1.0 if pred_sev == gt_sev else 0.0,
        "Severity_MAE": abs(pred_sev - gt_sev),
        "pred_area": pred_area,
        "gt_area": gt_area,
    }


def aggregate_measurement_metrics(samples):
    samples = list(samples)
    if not samples:
        raise ValueError("No samples provided for measurement evaluation")
    keys = ["RAE", "LE", "BDE", "CE", "ConnE", "Severity_Acc", "Severity_MAE"]
    result = {key: float(np.mean([sample[key] for sample in samples])) for key in keys}
    result["Spearman_rho"] = spearman_rho(
        [sample["pred_area"] for sample in samples],
        [sample["gt_area"] for sample in samples],
    )
    result["samples"] = len(samples)
    return result


def _scan_masks(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES:
            key = path.relative_to(root).with_suffix("").as_posix()
            files[key] = path
    return files


def evaluate_measurement_dirs(pred_dir, gt_dir, severity_thresholds=(0.005, 0.02)):
    pred_files = _scan_masks(pred_dir)
    gt_files = _scan_masks(gt_dir)
    keys = sorted(set(pred_files) & set(gt_files))
    if not keys:
        raise FileNotFoundError("No matching mask files found between {} and {}".format(pred_dir, gt_dir))
    samples = []
    for key in keys:
        pred = load_binary_mask(pred_files[key])
        gt = load_binary_mask(gt_files[key])
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0
        samples.append(sample_measurement_metrics(pred, gt, severity_thresholds=severity_thresholds))
    return aggregate_measurement_metrics(samples)

