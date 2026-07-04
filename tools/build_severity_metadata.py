import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from metrics.severity_metrics import build_metadata_from_masks, save_metadata


MASK_SUFFIXES = [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]


def load_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def find_mask(mask_root, sample_id):
    mask_root = Path(mask_root)
    sample = sample_id.replace("\\", "/")
    candidates = []
    stem = Path(sample)
    if stem.suffix:
        candidates.append(mask_root / sample)
    else:
        for suffix in MASK_SUFFIXES:
            candidates.append(mask_root / (sample + suffix))
            candidates.append(mask_root / (sample + "_GT" + suffix))
    for item in candidates:
        if item.is_file():
            return item
    raise FileNotFoundError("No mask found for sample '{}' under {}".format(sample_id, mask_root))


def read_split(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build severity metadata from training GT masks only.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split_file = Path(args.split_file)
    sample_ids = read_split(split_file)
    masks = [load_mask(find_mask(args.mask_root, sample_id)) for sample_id in sample_ids]
    metadata = build_metadata_from_masks(
        masks,
        dataset_name=args.dataset_name,
        split_file=str(split_file),
        sample_ids=sample_ids,
    )
    metadata["split_sha256"] = hashlib.sha256(split_file.read_bytes()).hexdigest()
    save_metadata(metadata, args.out)
    print("Saved severity metadata to {}".format(args.out))


if __name__ == "__main__":
    main()
