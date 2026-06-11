import argparse

from metrics.segmentation_metrics import evaluate_mask_dirs
from common import metric_text, write_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation masks.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted binary masks.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth binary masks.")
    parser.add_argument("--output", default=None, help="Optional output txt path.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_mask_dirs(args.pred_dir, args.gt_dir)
    print(metric_text(metrics))
    if args.output:
        write_metrics(args.output, metrics)
        print("Saved metrics to {}".format(args.output))


if __name__ == "__main__":
    main()

