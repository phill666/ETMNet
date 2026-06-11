from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


MASK_SUFFIXES = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}


class BinarySegmentationMetrics:
    def __init__(self, eps=1e-7):
        self.eps = float(eps)
        self.confusion = torch.zeros((2, 2), dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, target):
        pred = torch.as_tensor(pred).detach().view(-1).cpu().long().clamp(0, 1)
        target = torch.as_tensor(target).detach().view(-1).cpu().long()
        keep = (target >= 0) & (target < 2)
        pred = pred[keep]
        target = target[keep]
        bins = torch.bincount(target * 2 + pred, minlength=4).reshape(2, 2).double()
        self.confusion += bins

    def compute(self):
        cm = self.confusion
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        eps = self.eps
        iou = tp / (tp + fp + fn + eps)
        defect_tp = tp[1]
        defect_fp = fp[1]
        defect_fn = fn[1]
        precision = defect_tp / (defect_tp + defect_fp + eps)
        recall = defect_tp / (defect_tp + defect_fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        dice = 2.0 * defect_tp / (2.0 * defect_tp + defect_fp + defect_fn + eps)
        pixel_acc = tp.sum() / (cm.sum() + eps)
        return {
            "mIoU": float(iou.mean().item()),
            "background_IoU": float(iou[0].item()),
            "defect_IoU": float(iou[1].item()),
            "Dice": float(dice.item()),
            "F1": float(f1.item()),
            "Precision": float(precision.item()),
            "Recall": float(recall.item()),
            "Pixel_Accuracy": float(pixel_acc.item()),
        }


def logits_to_prediction(logits, threshold=None):
    if threshold is None:
        return torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)[:, 1]
    return (probs >= float(threshold)).long()


@torch.no_grad()
def update_from_logits(metrics, logits, target, threshold=None):
    if logits.shape[2:] != target.shape[1:]:
        logits = F.interpolate(logits, size=target.shape[1:], mode="bilinear", align_corners=False)
    pred = logits_to_prediction(logits, threshold=threshold)
    metrics.update(pred, target)
    return pred


def load_binary_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def _scan_masks(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES:
            key = path.relative_to(root).with_suffix("").as_posix()
            files[key] = path
    return files


def evaluate_mask_dirs(pred_dir, gt_dir):
    pred_files = _scan_masks(pred_dir)
    gt_files = _scan_masks(gt_dir)
    keys = sorted(set(pred_files) & set(gt_files))
    if not keys:
        raise FileNotFoundError("No matching mask files found between {} and {}".format(pred_dir, gt_dir))
    metrics = BinarySegmentationMetrics()
    for key in keys:
        pred = load_binary_mask(pred_files[key])
        gt = load_binary_mask(gt_files[key])
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 0
        metrics.update(pred.astype(np.uint8), gt.astype(np.uint8))
    result = metrics.compute()
    result["samples"] = len(keys)
    return result

