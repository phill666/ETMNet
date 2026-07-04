import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import metric_text, write_metrics
from metrics.measurement_metrics import evaluate_measurement_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate geometry-oriented defect metrics under the ETMNet revision protocol.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted binary masks.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth binary masks.")
    parser.add_argument("--severity-metadata", default=None, help="Training-GT severity metadata JSON.")
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument(
        "--severity-thresholds",
        nargs=2,
        type=float,
        default=(0.005, 0.02),
        metavar=("LOW", "HIGH"),
        help="Deprecated compatibility option; ignored by the revision protocol.",
    )
    parser.add_argument("--output", default=None, help="Optional output txt path.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_measurement_dirs(
        args.pred_dir,
        args.gt_dir,
        severity_metadata_path=args.severity_metadata,
        min_component_area=args.min_component_area,
        severity_thresholds=tuple(args.severity_thresholds),
    )
    print(metric_text(metrics))
    if args.output:
        write_metrics(args.output, metrics)
        print("Saved metrics to {}".format(args.output))


if __name__ == "__main__":
    main()
