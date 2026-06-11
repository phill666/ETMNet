import argparse

import torch

from common import (
    build_dataset,
    build_model,
    evaluate_loader,
    load_config,
    load_model_weights,
    make_loader,
    metric_text,
    repo_path,
    set_seed,
    write_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test ETMNet.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to ETMNet checkpoint.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional foreground probability threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = build_dataset(config, "test", is_train=False)
    test_loader = make_loader(
        test_dataset,
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

    metrics = evaluate_loader(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        use_amp=bool(config.get("amp", True)) and device.type == "cuda",
    )
    save_dir = repo_path(config.get("save_dir", "runs/etmnet"))
    suffix = "threshold_{:.3f}".format(args.threshold).replace(".", "p") if args.threshold is not None else "argmax"
    output_path = save_dir / "test_metrics_{}.txt".format(suffix)
    write_metrics(output_path, metrics)
    print(metric_text(metrics))
    print("Saved metrics to {}".format(output_path))


if __name__ == "__main__":
    main()

