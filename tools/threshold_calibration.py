import argparse
import csv
import math

import torch
import torch.nn.functional as F

from common import (
    build_dataset,
    build_model,
    load_config,
    load_model_weights,
    make_loader,
    metric_text,
    parse_hw,
    repo_path,
    save_yaml,
    set_seed,
)
from metrics.measurement_metrics import aggregate_measurement_metrics, sample_measurement_metrics
from metrics.segmentation_metrics import BinarySegmentationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-based threshold calibration for ETMNet.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to ETMNet checkpoint.")
    parser.add_argument("--mode", choices=("dice", "measurement", "both"), default="both")
    parser.add_argument("--save-dir", default=None, help="Directory for calibration outputs.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None, help="Override threshold list.")
    return parser.parse_args()


def main_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


@torch.no_grad()
def evaluate_threshold(model, loader, device, threshold, input_hw):
    model.eval()
    seg_metrics = BinarySegmentationMetrics()
    measurement_samples = []
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = main_logits(model(images))
        if logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= float(threshold)).long()
        seg_metrics.update(preds, masks)
        for pred_np, gt_np in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            measurement_samples.append(sample_measurement_metrics(pred_np, gt_np))
    seg = seg_metrics.compute()
    measurement = aggregate_measurement_metrics(measurement_samples)
    input_h, input_w = input_hw
    diag = math.hypot(input_h, input_w)
    measurement_score = (
        measurement["RAE"]
        + measurement["LE"] / max(input_h, input_w, 1)
        + measurement["BDE"] / max(diag, 1.0)
        + measurement["ConnE"]
        + measurement["Severity_MAE"]
    )
    return seg, measurement, float(measurement_score)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = args.thresholds or config.get("thresholds") or [x / 100.0 for x in range(20, 81, 5)]
    thresholds = [float(value) for value in thresholds]

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
        print("WARNING missing checkpoint keys: {}".format(msg.missing_keys))
    if msg.unexpected_keys:
        print("WARNING unexpected checkpoint keys: {}".format(msg.unexpected_keys))

    save_dir = repo_path(args.save_dir or (str(config.get("save_dir", "runs/etmnet")) + "_thresholds"))
    save_dir.mkdir(parents=True, exist_ok=True)
    input_hw = parse_hw(config.get("input_size", [512, 512]))

    rows = []
    for threshold in thresholds:
        seg, measurement, measurement_score = evaluate_threshold(model, val_loader, device, threshold, input_hw)
        row = {"threshold": threshold, **seg, **measurement, "measurement_score": measurement_score}
        rows.append(row)
        print(
            "threshold={:.3f} Dice={:.6f} defect_IoU={:.6f} measurement_score={:.6f}".format(
                threshold, seg["Dice"], seg["defect_IoU"], measurement_score
            )
        )

    dice_row = max(rows, key=lambda item: (item["Dice"], item["defect_IoU"]))
    measurement_row = min(rows, key=lambda item: item["measurement_score"])
    calibration = {
        "DiceOpt": float(dice_row["threshold"]),
        "MeasurementOpt": float(measurement_row["threshold"]),
        "selection_mode": args.mode,
        "diceopt_metrics": {k: v for k, v in dice_row.items() if k != "threshold"},
        "measurementopt_metrics": {k: v for k, v in measurement_row.items() if k != "threshold"},
    }

    fieldnames = list(rows[0].keys())
    csv_path = save_dir / "threshold_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    yaml_path = save_dir / "best_thresholds.yaml"
    save_yaml(calibration, yaml_path)

    print("DiceOpt threshold: {:.3f}".format(calibration["DiceOpt"]))
    print("MeasurementOpt threshold: {:.3f}".format(calibration["MeasurementOpt"]))
    if args.mode == "dice":
        print("Selected DiceOpt")
        print(metric_text({k: v for k, v in dice_row.items() if k != "threshold"}))
    elif args.mode == "measurement":
        print("Selected MeasurementOpt")
        print(metric_text({k: v for k, v in measurement_row.items() if k != "threshold"}))
    else:
        print("Saved both DiceOpt and MeasurementOpt")
    print("Saved calibration CSV to {}".format(csv_path))
    print("Saved best thresholds to {}".format(yaml_path))


if __name__ == "__main__":
    main()

