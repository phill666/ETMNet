import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from common import build_dataset, build_model, load_config, load_model_weights, make_loader, repo_path, set_seed


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Save ETMNet prediction masks and overlays.")
    parser.add_argument("--config", required=True, help="Path to ETMNet YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to ETMNet checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for masks and overlays.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=20)
    return parser.parse_args()


def main_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def tensor_to_image(image_tensor):
    array = image_tensor.detach().cpu().numpy()
    array = np.clip((array * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def overlay_mask(image_np, mask_np):
    overlay = image_np.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    mask = mask_np.astype(bool)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * red[mask]).astype(np.uint8)
    return overlay


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, args.split, is_train=False)
    loader = make_loader(dataset, batch_size=1, num_workers=int(config.get("num_workers", 4)), shuffle=False, device=device)
    model = build_model(config).to(device).eval()
    msg = load_model_weights(model, args.checkpoint, device)
    if msg.missing_keys:
        print("WARNING missing checkpoint keys: {}".format(msg.missing_keys))
    if msg.unexpected_keys:
        print("WARNING unexpected checkpoint keys: {}".format(msg.unexpected_keys))

    output_dir = repo_path(args.output_dir)
    pred_dir = output_dir / "pred"
    overlay_dir = output_dir / "overlay"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for images, masks, keys in loader:
        images = images.to(device, non_blocking=True)
        logits = main_logits(model(images))
        if logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= float(args.threshold)).cpu().numpy().astype(np.uint8)
        for i, key in enumerate(keys):
            clean_key = Path(str(key)).with_suffix("").as_posix()
            pred_path = pred_dir / "{}.png".format(clean_key)
            overlay_path = overlay_dir / "{}.png".format(clean_key)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            pred_np = preds[i] * 255
            image_np = tensor_to_image(images[i])
            Image.fromarray(pred_np, mode="L").save(pred_path)
            Image.fromarray(overlay_mask(image_np, preds[i])).save(overlay_path)
            saved += 1
            if saved >= int(args.max_samples):
                print("Saved {} prediction visualizations to {}".format(saved, output_dir))
                return
    print("Saved {} prediction visualizations to {}".format(saved, output_dir))


if __name__ == "__main__":
    main()

