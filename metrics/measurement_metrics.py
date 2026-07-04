"""Official measurement metrics for the ETMNet revision protocol.

This file is the public default entry point. The previous implementation is
retained in ``metrics/legacy_measurement_metrics.py`` only for backward
compatibility and is not used by the revised protocol.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .geometry_metrics import aggregate_geometry_records, as_binary, sample_geometry_record
from .severity_metrics import aggregate_severity, load_metadata, severity_record


MASK_SUFFIXES = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_binary_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def sample_measurement_metrics(pred, gt, severity_metadata=None, min_component_area=1):
    """Return per-sample measurement metrics.

    RAE, LE, BDE, CE, and ConnE are valid only for GT-positive samples. For a
    GT-positive complete miss, BDE and CE use image-diagonal penalties. CE is
    centroid distance in original-image pixels; ConnE is the absolute
    eight-neighbor connected-component count difference.
    """
    pred = as_binary(pred)
    gt = as_binary(gt)
    record = sample_geometry_record(pred, gt, min_component_area=min_component_area)
    out = {
        "RAE": record["rae"],
        "LE": record["le"],
        "BDE": record["bde"],
        "CE": record["ce"],
        "ConnE": record["conne"],
        "pred_area": record["pred_area"],
        "gt_area": record["gt_area"],
        "prediction_empty": record["prediction_empty"],
    }
    if severity_metadata is not None and record["gt_area"] > 0:
        out.update(severity_record(pred, gt, severity_metadata))
    return out


def aggregate_measurement_metrics(samples, severity_metadata=None):
    samples = list(samples)
    if not samples:
        raise ValueError("No samples provided for measurement evaluation.")
    geometry = aggregate_geometry_records(samples)
    severity_rows = []
    if severity_metadata is not None:
        for item in samples:
            if item.get("gt_area", 0) > 0 and {"pred_score", "gt_score", "pred_label", "gt_label"} <= set(item):
                severity_rows.append(item)
    if severity_metadata is None:
        severity = {"Severity_Accuracy": "-", "Severity_MAE": "-", "Spearman": "-"}
    else:
        severity = aggregate_severity(severity_rows)
    return {**geometry, **severity}


def _scan_masks(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES:
            key = path.relative_to(root).with_suffix("").as_posix()
            files[key] = path
    return files


def evaluate_measurement_dirs(pred_dir, gt_dir, severity_metadata_path=None, min_component_area=1, severity_thresholds=None):
    """Evaluate paired prediction and GT mask directories.

    ``severity_thresholds`` is accepted only for old caller compatibility and is
    ignored by the revised protocol. Pass ``severity_metadata_path`` generated
    from training GT masks to enable severity metrics.
    """
    pred_files = _scan_masks(pred_dir)
    gt_files = _scan_masks(gt_dir)
    keys = sorted(set(pred_files) & set(gt_files))
    if not keys:
        raise FileNotFoundError("No matching mask files found between {} and {}".format(pred_dir, gt_dir))
    metadata = load_metadata(severity_metadata_path) if severity_metadata_path else None
    samples = []
    for key in keys:
        pred = load_binary_mask(pred_files[key])
        gt = load_binary_mask(gt_files[key])
        if pred.shape != gt.shape:
            pred = (np.asarray(Image.fromarray(pred * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0).astype(np.uint8)
        rec = sample_geometry_record(pred, gt, min_component_area=min_component_area)
        if metadata is not None and rec["gt_area"] > 0:
            rec.update(severity_record(pred, gt, metadata))
        samples.append(rec)
    return aggregate_measurement_metrics(samples, severity_metadata=metadata)
