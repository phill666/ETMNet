import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from common import (
    build_dataset,
    build_model,
    evaluate_loader,
    load_config,
    make_loader,
    metric_text,
    repo_path,
    save_yaml,
    set_seed,
)
from losses import ETMNetLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train ETMNet.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    return parser.parse_args()


def poly_lr(base_lr, current_iter, max_iter, power=0.9):
    return float(base_lr) * (1.0 - float(current_iter) / max(float(max_iter), 1.0)) ** float(power)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr) * float(group.get("lr_mult", 1.0))


def build_optimizer(config, model):
    name = str(config.get("optimizer", "SGD")).lower()
    lr = float(config.get("learning_rate", 0.01))
    weight_decay = float(config.get("weight_decay", 0.0005))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError("Unsupported optimizer: {}".format(config.get("optimizer")))


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = repo_path(config.get("save_dir", "runs/etmnet"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, save_dir / "config_used.yaml")

    train_dataset = build_dataset(config, "train", is_train=True)
    val_dataset = build_dataset(config, "val", is_train=False)
    train_loader = make_loader(
        train_dataset,
        batch_size=int(config.get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 4)),
        shuffle=True,
        device=device,
        seed=seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        num_workers=int(config.get("num_workers", 4)),
        shuffle=False,
        device=device,
        seed=seed,
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    criterion = ETMNetLoss(
        ce_weight=float(config.get("ce_weight", 1.0)),
        dice_weight=float(config.get("dice_weight", 1.0)),
        measurement_dice_weight=float(config.get("measurement_dice_weight", 1.0)),
        aux_weight=float(config.get("aux_weight", 0.4)),
        boundary_weight=float(config.get("boundary_weight", 0.2)),
        use_measurement_dice=bool(config.get("use_measurement_dice", True)),
        area_gamma=float(config.get("measurement_area_gamma", 0.5)),
    )

    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(config.get("epochs", 120))
    max_iter = max(1, epochs * len(train_loader))
    base_lr = float(config.get("learning_rate", 0.01))
    best_defect_iou = -1.0
    global_iter = 0
    best_path = save_dir / "best_etmnet.pth"
    log_path = save_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "val_mIoU",
                "val_defect_IoU",
                "val_Dice",
                "val_F1",
                "val_Precision",
                "val_Recall",
                "best_defect_IoU",
            ],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0
            progress = tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epochs))
            current_lr = base_lr
            for images, masks, _ in progress:
                current_lr = poly_lr(base_lr, global_iter, max_iter, power=float(config.get("power", 0.9)))
                set_optimizer_lr(optimizer, current_lr)
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images, return_extra=True)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = images.size(0)
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                global_iter += 1
                progress.set_postfix(loss="{:.4f}".format(total_loss / max(total_samples, 1)))

            train_loss = total_loss / max(total_samples, 1)
            val_metrics = evaluate_loader(model, val_loader, device, threshold=None, use_amp=use_amp)
            if val_metrics["defect_IoU"] > best_defect_iou:
                best_defect_iou = val_metrics["defect_IoU"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_defect_iou": best_defect_iou,
                        "metrics": val_metrics,
                        "config": config,
                    },
                    best_path,
                )

            row = {
                "epoch": epoch,
                "lr": "{:.8f}".format(current_lr),
                "train_loss": "{:.6f}".format(train_loss),
                "val_mIoU": "{:.6f}".format(val_metrics["mIoU"]),
                "val_defect_IoU": "{:.6f}".format(val_metrics["defect_IoU"]),
                "val_Dice": "{:.6f}".format(val_metrics["Dice"]),
                "val_F1": "{:.6f}".format(val_metrics["F1"]),
                "val_Precision": "{:.6f}".format(val_metrics["Precision"]),
                "val_Recall": "{:.6f}".format(val_metrics["Recall"]),
                "best_defect_IoU": "{:.6f}".format(best_defect_iou),
            }
            writer.writerow(row)
            f.flush()
            print("epoch={} train_loss={:.6f}".format(epoch, train_loss))
            print(metric_text(val_metrics))
            print("best_checkpoint: {}".format(best_path))

    print("Training finished. Best checkpoint: {}".format(best_path))


if __name__ == "__main__":
    main()

