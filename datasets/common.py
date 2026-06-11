import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset


try:
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


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
    raise ValueError("input_size must be an int or a pair of integers")


def _relative_key(path, root):
    rel = Path(path).relative_to(root)
    return rel.with_suffix("").as_posix()


def _scan_files(root, suffixes):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError("Directory does not exist: {}".format(root))
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = _relative_key(path, root)
        if key in files:
            raise ValueError("Duplicate sample key '{}' under {}".format(key, root))
        files[key] = path
    return files


def _read_split(split_file):
    if split_file is None:
        return None
    with Path(split_file).open("r", encoding="utf-8") as f:
        values = [line.strip() for line in f if line.strip()]
    keys = []
    for value in values:
        path = Path(value.replace("\\", "/"))
        keys.append(path.with_suffix("").as_posix())
    return keys


def _resolve_keys(requested_keys, image_files, mask_files):
    if requested_keys is None:
        requested_keys = sorted(set(image_files) & set(mask_files))
    samples = []
    missing = []
    for key in requested_keys:
        image = image_files.get(key)
        mask = mask_files.get(key)
        if image is None or mask is None:
            missing.append(key)
            continue
        samples.append({"key": key, "image": image, "mask": mask})
    if missing:
        preview = ", ".join(missing[:20])
        raise FileNotFoundError("Split references missing image/mask pairs: {}".format(preview))
    return samples


def _binary_mask(mask):
    array = np.asarray(mask.convert("L"))
    return Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L")


class BinaryDefectDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        split_file=None,
        input_size=(512, 512),
        is_train=True,
        scale_range=(0.75, 1.5),
        hflip_prob=0.5,
        vflip_prob=0.5,
        color_jitter=(0.1, 0.1),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_h, self.input_w = parse_hw(input_size)
        self.is_train = bool(is_train)
        self.scale_range = tuple(scale_range)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.color_jitter = tuple(color_jitter)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        image_files = _scan_files(self.image_dir, IMAGE_SUFFIXES)
        mask_files = _scan_files(self.mask_dir, MASK_SUFFIXES)
        requested_keys = _read_split(split_file)
        self.samples = _resolve_keys(requested_keys, image_files, mask_files)

    def __len__(self):
        return len(self.samples)

    def _resize(self, image, mask):
        size = (self.input_w, self.input_h)
        return image.resize(size, BILINEAR), mask.resize(size, NEAREST)

    def _random_scale(self, image, mask):
        scale = random.uniform(float(self.scale_range[0]), float(self.scale_range[1]))
        width, height = image.size
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return image.resize(size, BILINEAR), mask.resize(size, NEAREST)

    def _random_crop(self, image, mask):
        crop_w, crop_h = self.input_w, self.input_h
        width, height = image.size
        pad_w = max(0, crop_w - width)
        pad_h = max(0, crop_h - height)
        if pad_w or pad_h:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)
            width, height = image.size
        left = 0 if width == crop_w else random.randint(0, width - crop_w)
        top = 0 if height == crop_h else random.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        return image.crop(box), mask.crop(box)

    def _augment(self, image, mask):
        image, mask = self._random_scale(image, mask)
        image, mask = self._random_crop(image, mask)
        if random.random() < self.hflip_prob:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        if random.random() < self.vflip_prob:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        brightness, contrast = self.color_jitter
        if brightness > 0:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(1.0 - brightness, 1.0 + brightness))
        if contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(1.0 - contrast, 1.0 + contrast))
        return image, mask

    def _to_tensors(self, image, mask):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_np = image_np.transpose(2, 0, 1).copy()
        mask_np = (np.asarray(mask) > 0).astype(np.int64)
        return torch.from_numpy(image_np).float(), torch.from_numpy(mask_np).long()

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image"]).convert("RGB")
        mask = _binary_mask(Image.open(sample["mask"]))
        if self.is_train:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._resize(image, mask)
        image_tensor, mask_tensor = self._to_tensors(image, mask)
        return image_tensor, mask_tensor, sample["key"]

