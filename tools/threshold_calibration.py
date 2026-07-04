import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv

import torch
import torch.nn.functional as F

from common import (
    build_dataset,
    build_model,
    load_config,
    load_model_weights,
    make_loader,
    metric_text,
    repo_path,
    save_yaml,
    set_seed,
)
from metrics.geometry_metrics import sample_geometry_record
from metrics.measurement_metrics import aggregate_measurement_metrics
from metrics.segmentation_metrics import BinarySegmentationMetrics
from metrics.threshold_calibration import THRESHOLD_CANDIDATES, add_measurement_normalization, select_diceopt, select_measurementopt


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-based threshold calibration for ETMNet revision protocol.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to ETMNet checkpoint.")
    parser.add_argument("--mode", choices=("dice", "measurement", "both"), default="both")
    parser.add_argument("--save-dir", default=None, help="Directory for calibration outputs.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None, help="Override threshold list.")
    parser.add_argument("--min-component-area", type=int, default=1)
    return parser.parse_args()


def main_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs.get("out", next(iter(outputs.values())))
    return outputs


@torch.no_grad()
def evaluate_threshold(model, loader, device, threshold, min_component_area=1):
    model.eval()
    seg_metrics = BinarySegmentationMetrics()
    measurement_samples = []
    dataset = getattr(loader, "dataset", None)
    use_original_masks = hasattr(dataset, "load_original_mask")
    for images, masks, keys in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = main_logits(model(images))
        if use_original_masks:
            for index, key in enumerate(keys):
                gt_np = dataset.load_original_mask(key)
                sample_logits = logits[index:index + 1]
                if sample_logits.shape[2:] != gt_np.shape:
                    sample_logits = F.interpolate(sample_logits, size=gt_np.shape, mode="bilinear", align_corners=True)
                prob = torch.softmax(sample_logits, dim=1)[:, 1][0]
                pred_np = (prob >= float(threshold)).cpu().numpy().astype("uint8")
                seg_metrics.update(pred_np, gt_np)
                measurement_samples.append(sample_geometry_record(pred_np, gt_np, min_component_area=min_component_area))
        else:
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=True)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= float(threshold)).long()
            seg_metrics.update(preds, masks)
            for pred_np, gt_np in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                measurement_samples.append(sample_geometry_record(pred_np, gt_np, min_component_area=min_component_area))
    seg = seg_metrics.compute()
    measurement = aggregate_measurement_metrics(measurement_samples)
    return seg, measurement

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [float(value) for value in (args.thresholds or config.get("thresholds") or THRESHOLD_CANDIDATES)]

    val_dataset = build_dataset(config, "val", is_train=False)
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        num_workers=int(config.get("num_workers", 4)),
        shuffle=False,
        device=device,
        seed=int(config.get("seed", 42)),
    )
    model = build_model(config).to(device)
    msg = load_model_weights(model, args.checkpoint, device)
    if msg.missing_keys:
        raise RuntimeError("Missing checkpoint keys: {}".format(msg.missing_keys))

    save_dir = repo_path(args.save_dir or (str(config.get("save_dir", "runs/etmnet")) + "_thresholds"))
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selection_rows = []
    for threshold in thresholds:
        seg, measurement = evaluate_threshold(model, val_loader, device, threshold, min_component_area=args.min_component_area)
        row = {"threshold": threshold, **seg, **measurement}
        rows.append(row)
        selection_rows.append({"threshold": threshold, "Dice/F1": seg["Dice"], "RAE": measurement.get("RAE"), "BDE": measurement.get("BDE"), "CE": measurement.get("CE")})
        print("threshold={:.3f} Dice={:.6f} defect_IoU={:.6f} RAE={} BDE={} CE={}".format(threshold, seg["Dice"], seg["defect_IoU"], measurement.get("RAE"), measurement.get("BDE"), measurement.get("CE")))

    normalized = add_measurement_normalization(selection_rows)
    for row, norm in zip(rows, normalized):
        row["normalized_RAE"] = norm.get("normalized_RAE", "")
        row["normalized_BDE"] = norm.get("normalized_BDE", "")
        row["normalized_CE"] = norm.get("normalized_CE", "")
        row["measurement_score"] = norm.get("measurement_score", "")

    dice_selected = select_diceopt(selection_rows)
    measurement_selected = select_measurementopt(selection_rows)
    dice_row = next(row for row in rows if float(row["threshold"]) == float(dice_selected["threshold"]))
    measurement_row = next(row for row in rows if float(row["threshold"]) == float(measurement_selected["threshold"]))
    calibration = {
        "protocol_version": "revision_2026_07",
        "DiceOpt": float(dice_row["threshold"]),
        "MeasurementOpt": float(measurement_row["threshold"]),
        "selection_mode": args.mode,
        "threshold_candidates": thresholds,
        "diceopt_metrics": {k: v for k, v in dice_row.items() if k != "threshold"},
        "measurementopt_metrics": {k: v for k, v in measurement_row.items() if k != "threshold"},
    }

    fieldnames = list(rows[0].keys())
    csv_path = save_dir / "threshold_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    yaml_path = save_dir / "best_thresholds.yaml"
    save_yaml(calibration, yaml_path)

    print("DiceOpt threshold: {:.3f}".format(calibration["DiceOpt"]))
    print("MeasurementOpt threshold: {:.3f}".format(calibration["MeasurementOpt"]))
    if args.mode == "dice":
        print(metric_text({k: v for k, v in dice_row.items() if k != "threshold"}))
    elif args.mode == "measurement":
        print(metric_text({k: v for k, v in measurement_row.items() if k != "threshold"}))
    else:
        print("Saved both DiceOpt and MeasurementOpt")
    print("Saved calibration CSV to {}".format(csv_path))
    print("Saved best thresholds to {}".format(yaml_path))


if __name__ == "__main__":
    main()
