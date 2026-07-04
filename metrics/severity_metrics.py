"""Severity metadata and metrics for ETMNet revision protocol."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry_metrics import area, boundary_irregularity, skeleton_length


PROTOCOL_VERSION = "revision_2026_07"
EPS = 1e-6


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def minmax_norm(value, lo, hi):
    if abs(float(hi) - float(lo)) < EPS:
        return 0.0
    return _clip01((float(value) - float(lo)) / (float(hi) - float(lo)))


def mask_features(mask):
    return {
        "area": float(area(mask)),
        "length": float(skeleton_length(mask)),
        "boundary_irregularity": float(boundary_irregularity(mask)),
    }


def severity_score_from_features(features, metadata):
    stats = metadata["stats"]
    a = minmax_norm(features["area"], stats["A_min_train"], stats["A_max_train"])
    l = minmax_norm(features["length"], stats["L_min_train"], stats["L_max_train"])
    b = minmax_norm(features["boundary_irregularity"], stats["B_min_train"], stats["B_max_train"])
    return float((a + l + b) / 3.0)


def severity_label(score, metadata):
    q1 = float(metadata["quantiles"]["q33_3"])
    q2 = float(metadata["quantiles"]["q66_7"])
    if score <= q1:
        return 0
    if score <= q2:
        return 1
    return 2


def build_metadata_from_masks(masks, dataset_name="", split_file="", sample_ids=None):
    masks = list(masks)
    sample_ids = list(sample_ids) if sample_ids is not None else [str(i) for i in range(len(masks))]
    if not masks:
        raise ValueError("No training masks were provided.")
    feats = [mask_features(mask) for mask in masks if area(mask) > 0]
    if not feats:
        raise ValueError("Severity metadata requires at least one GT-positive training mask.")
    areas = np.asarray([f["area"] for f in feats], dtype=np.float64)
    lens = np.asarray([f["length"] for f in feats], dtype=np.float64)
    irrs = np.asarray([f["boundary_irregularity"] for f in feats], dtype=np.float64)
    stats = {
        "A_min_train": float(areas.min()),
        "A_max_train": float(areas.max()),
        "L_min_train": float(lens.min()),
        "L_max_train": float(lens.max()),
        "B_min_train": float(irrs.min()),
        "B_max_train": float(irrs.max()),
    }
    tmp = {"stats": stats, "quantiles": {"q33_3": 0.0, "q66_7": 1.0}}
    scores = np.asarray([severity_score_from_features(f, tmp) for f in feats], dtype=np.float64)
    q1, q2 = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    split_hash = ""
    if split_file and Path(split_file).is_file():
        split_hash = hashlib.sha256(Path(split_file).read_bytes()).hexdigest()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": dataset_name,
        "split_file": str(split_file),
        "split_sha256": split_hash,
        "sample_count": len(masks),
        "gt_positive_sample_count": len(feats),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "quantiles": {"q33_3": float(q1), "q66_7": float(q2)},
    }


def load_metadata(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def severity_record(pred_mask, gt_mask, metadata):
    pred_features = mask_features(pred_mask)
    gt_features = mask_features(gt_mask)
    pred_score = severity_score_from_features(pred_features, metadata)
    gt_score = severity_score_from_features(gt_features, metadata)
    return {
        "pred_score": pred_score,
        "gt_score": gt_score,
        "pred_label": severity_label(pred_score, metadata),
        "gt_label": severity_label(gt_score, metadata),
    }


def _rankdata(values):
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks


def spearman_rho(x, y):
    if len(x) < 2:
        return "-"
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return "-"
    value = float(np.corrcoef(rx, ry)[0, 1])
    return "-" if np.isnan(value) else value


def aggregate_severity(records):
    records = list(records)
    if not records:
        return {"Severity_Accuracy": "-", "Severity_MAE": "-", "Spearman": "-"}
    pred_labels = np.asarray([r["pred_label"] for r in records])
    gt_labels = np.asarray([r["gt_label"] for r in records])
    pred_scores = np.asarray([r["pred_score"] for r in records], dtype=np.float64)
    gt_scores = np.asarray([r["gt_score"] for r in records], dtype=np.float64)
    return {
        "Severity_Accuracy": float(np.mean(pred_labels == gt_labels)),
        "Severity_MAE": float(np.mean(np.abs(pred_scores - gt_scores))),
        "Spearman": spearman_rho(gt_scores, pred_scores),
    }
