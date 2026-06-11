import argparse

from common import metric_text, write_metrics
from metrics.measurement_metrics import evaluate_measurement_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate measurement-oriented defect metrics.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted binary masks.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth binary masks.")
    parser.add_argument(
        "--severity-thresholds",
        nargs=2,
        type=float,
        default=(0.005, 0.02),
        metavar=("LOW", "HIGH"),
        help="Area-ratio thresholds for low/medium/high severity.",
    )
    parser.add_argument("--output", default=None, help="Optional output txt path.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_measurement_dirs(
        args.pred_dir,
        args.gt_dir,
        severity_thresholds=tuple(args.severity_thresholds),
    )
    print(metric_text(metrics))
    if args.output:
        write_metrics(args.output, metrics)
        print("Saved metrics to {}".format(args.output))


if __name__ == "__main__":
    main()

