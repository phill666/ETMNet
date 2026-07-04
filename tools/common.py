import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import KolektorSDD2Dataset, MagneticTileDataset, NEUSegDataset
from metrics.segmentation_metrics import BinarySegmentationMetrics, logits_to_prediction, update_from_logits
from models import ETMNet


def repo_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists() or path.parent.exists():
        return path
    return REPO_ROOT / path


def load_config(path):
    path = repo_path(path)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read config files") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data, path):
    import yaml

    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_hw(value):
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
        if len(parts) == 1:
            return int(parts[0]), int(parts[0])
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("input_size must be an int or two integers")


def dataset_class(dataset_name):
    normalized = str(dataset_name).lower().replace("-", "_").replace(" ", "_")
    if normalized in {"neu_seg", "neuseg"}:
        return NEUSegDataset
    if normalized in {"kolektorsdd2", "kolektor_sdd2", "ksdd2"}:
        return KolektorSDD2Dataset
    if normalized in {"magnetic_tile_defect", "magnetic_tile", "mtd"}:
        return MagneticTileDataset
    raise ValueError("Unsupported dataset_name: {}".format(dataset_name))


def build_dataset(config, split, is_train):
    cls = dataset_class(config["dataset_name"])
    split_file = config.get("{}_split_file".format(split))
    return cls(
        image_dir=repo_path(config["{}_image_dir".format(split)]),
        mask_dir=repo_path(config["{}_mask_dir".format(split)]),
        split_file=repo_path(split_file) if split_file else None,
        input_size=config.get("input_size", [512, 512]),
        is_train=is_train,
        image_suffixes=config.get("image_suffixes"),
        mask_suffixes=config.get("mask_suffixes"),
    )


def make_loader(dataset, batch_size, num_workers, shuffle, device, seed=42):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=bool(shuffle),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(num_workers) > 0,
    )


def build_model(config):
    model_name = str(config.get("model_name", "ETMNet")).lower()
    if model_name not in {"etmnet", "pidnet"}:
        raise ValueError("Unsupported model_name={}".format(config.get("model_name")))
    if model_name == "pidnet":
        if bool(config.get("use_edgeflowconv", False)):
            raise ValueError("PIDNet config must keep use_edgeflowconv=false.")
        if bool(config.get("use_topology_context", False)):
            raise ValueError("PIDNet config must keep use_topology_context=false.")
    return ETMNet(
        num_classes=int(config.get("num_classes", 2)),
        aux_loss=bool(config.get("aux_loss", True)),
        use_edgeflowconv=False if model_name == "pidnet" else bool(config.get("use_edgeflowconv", True)),
        use_topology_context=False if model_name == "pidnet" else bool(config.get("use_topology_context", True)),
        use_boundary_aux=float(config.get("boundary_weight", 0.0)) > 0,
        edge_gamma=float(config.get("edge_gamma", 0.3)),
        topology_gamma=float(config.get("topology_gamma", 0.2)),
    )


def main_logits(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 3 and isinstance(outputs[2], dict):
        return outputs[0]
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(repo_path(path), map_location=map_location)
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint
    state = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    return checkpoint, state


def load_model_weights(model, checkpoint_path, device):
    _, state = load_checkpoint(checkpoint_path, map_location=device)
    return model.load_state_dict(state, strict=False)


def metric_text(metrics):
    return "\n".join(
        "{}: {:.6f}".format(key, value) if isinstance(value, float) else "{}: {}".format(key, value)
        for key, value in metrics.items()
    )


def write_metrics(path, metrics):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metric_text(metrics) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_loader(model, loader, device, threshold=None, use_amp=False):
    model.eval()
    metrics = BinarySegmentationMetrics()
    dataset = getattr(loader, "dataset", None)
    use_original_masks = hasattr(dataset, "load_original_mask")
    for images, masks, keys in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        context = torch.cuda.amp.autocast(enabled=True) if bool(use_amp and device.type == "cuda") else nullcontext()
        with context:
            logits = main_logits(model(images))
        if use_original_masks:
            for index, key in enumerate(keys):
                gt_np = dataset.load_original_mask(key)
                sample_logits = logits[index:index + 1]
                if sample_logits.shape[2:] != gt_np.shape:
                    sample_logits = F.interpolate(sample_logits, size=gt_np.shape, mode="bilinear", align_corners=True)
                pred = logits_to_prediction(sample_logits, threshold=threshold)[0].cpu()
                metrics.update(pred, gt_np)
        else:
            update_from_logits(metrics, logits, masks, threshold=threshold)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return metrics.compute()