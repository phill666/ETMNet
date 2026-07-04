# ETMNet: An Edge-Guided Measurement Network for Real-Time Industrial Surface Defect Assessment

This repository provides the public ETMNet implementation and the revised evaluation protocol used for real-time industrial surface defect segmentation and geometry-oriented assessment.

## Overview

ETMNet is a lightweight segmentation model built on a real-time PIDNet-style backbone. Its core architectural addition is EdgeFlowConv, a directional boundary enhancement module for improving local boundary features. The revised paper positions ETMNet as an edge-guided measurement network, not as a structural-consistency optimizer.

## Method Scope

- Training uses cross-entropy plus Dice loss.
- Baseline PIDNet uses Dice weight `1.0`.
- ETMNet uses Dice weight `1.2`.
- Geometric metrics are used only for post-training evaluation and threshold analysis; they do not participate in gradient propagation or parameter updates.
- DiceOpt and MeasurementOpt are validation-set threshold calibration rules applied after training.
- ETMNet does not provide an explicit topology guarantee.
- Known limitations include ultra-thin scratches, low-contrast boundaries, and cases where overlap quality remains high while BDE or CE is still large.

## Repository Structure

```text
ETMNet_GitHub_Release/
|-- configs/                 # Dataset and revision calibration configs
|-- datasets/                # Dataset readers and preparation notes
|-- losses/                  # Cross-entropy + Dice loss helpers
|-- metrics/                 # Segmentation, geometry, severity, calibration metrics
|-- models/                  # ETMNet, EdgeFlowConv, and real-time backbone
|-- tools/                   # Training, evaluation, calibration, revision utilities
|-- tests/                   # Lightweight protocol tests
|-- PROTOCOL.md              # Revised evaluation protocol
|-- README.md
|-- CITATION.cff
|-- LICENSE
`-- requirements.txt
```

## Environment

```bash
conda create -n etmnet python=3.10 -y
conda activate etmnet
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the default wheel is not suitable for your machine. This repository does not install dependencies automatically.

## Dataset Preparation

Datasets are not included. Download NEU Seg, KolektorSDD2, and Magnetic Tile Defect from their official sources and keep fixed train/validation/test split files. A split file contains one sample key per line.

```text
data/
|-- NEU_Seg/
|-- KolektorSDD2/
`-- Magnetic_Tile_Defect/
```

The revision experiments used fixed splits. Do not select thresholds or severity metadata from the test split.

## Training

```bash
python tools/train.py --config configs/neu_seg_etmnet.yaml
python tools/train.py --config configs/kolektorsdd2_etmnet.yaml
python tools/train.py --config configs/magnetic_tile_etmnet.yaml
```

Training checkpoints are not included in this public release. The revised evaluation scripts can evaluate checkpoints produced by these configs.

## Standard Segmentation Evaluation

Segmentation metrics use dataset-level confusion-matrix accumulation: mIoU, background IoU, defect IoU, Dice/F1, Precision, Recall, and Pixel Accuracy. Normal samples are included in segmentation metrics. Binary masks use labels `0` for background and `1` for defects.

## DiceOpt Calibration

All models use the same candidate thresholds:

```text
0.05, 0.10, 0.15, ..., 0.95
```

DiceOpt is selected independently for each trained model on the validation set by maximizing validation Dice/F1. Ties are resolved by choosing the threshold closest to `0.50`; if still tied, the smaller threshold is selected.

## MeasurementOpt Calibration

MeasurementOpt is also selected only on the validation set. For each candidate threshold, compute validation `RAE`, `BDE`, and `CE`. For that model, min-max normalize each metric across the 19 candidates and minimize:

```text
measurement_score = normalized_RAE + normalized_BDE + normalized_CE
```

If a metric is constant across thresholds, its normalized value is set to `0` for all candidates. Ties use the same closest-to-`0.50`, then smaller-threshold rule. The selected threshold is then fixed for the test set.

## Geometric Metric Definitions

All geometric metrics are computed at the original image size after restoring the model probability map to the original resolution. The public revision scripts restore logits/probability maps with bilinear interpolation and `align_corners=True`. Ground-truth masks and binary prediction masks are never resized with bilinear interpolation.

- `RAE`: relative area error, `abs(pred_area - gt_area) / gt_area`, evaluated only for GT-positive samples.
- `LE`: relative skeleton length error. A binary defect mask is skeletonized, and the single-pixel skeleton is measured as an 8-neighbor graph: horizontal and vertical edges contribute `1`, diagonal edges contribute `sqrt(2)`, and each undirected edge is counted once.
- `BDE`: bidirectional boundary distance error. Boundaries are extracted as `mask - erode3x3(mask)` and distances are measured in original-image pixels.
- `CE`: Euclidean distance between predicted and GT defect centroids in original-image coordinates.
- `ConnE`: absolute difference between predicted and GT 8-neighbor connected component counts.

For GT-positive complete misses, predicted area, skeleton length, and component count are `0`; `BDE` and `CE` use the image diagonal as the penalty.

## Severity Metadata Generation

Severity metrics use dataset-level metadata generated only from training GT masks: area min/max, skeleton-length min/max, boundary-irregularity min/max, and the 33.3% and 66.7% quantiles of the continuous GT severity score.

The continuous severity score is:

```text
S = A_norm / 3 + L_norm / 3 + B_norm / 3
```

Generate metadata with:

```bash
python tools/build_severity_metadata.py --dataset-name Magnetic_Tile_Defect --mask-root data/Magnetic_Tile_Defect/masks --split-file data/Magnetic_Tile_Defect/splits/train.txt --out configs/severity_metadata/mtd.json
```

Only generate JSON files when the dataset and fixed split are available. Do not fill metadata with invented values.

## Empty-Mask Handling

RAE, LE, BDE, CE, ConnE, and severity metrics are averaged over GT-positive samples. GT-empty samples are retained for segmentation metrics and diagnostic counts: total samples, GT-positive samples, GT-empty samples, prediction-empty samples, and both-empty samples. GT-empty samples must not use tiny denominators to produce huge RAE values.

## Reproducing Revision Evaluation

```bash
python tools/threshold_calibration.py --config configs/magnetic_tile_etmnet.yaml --checkpoint <checkpoint.pth> --mode both
python tools/evaluate_measurement.py --pred-dir <pred-mask-dir> --gt-dir <gt-mask-dir> --severity-metadata configs/severity_metadata/mtd.json
```

Private checkpoints, raw datasets, and large revision output files are not included in this public repository.

Protocol validation:

```bash
python tools/validate_evaluation_protocol.py
python -m unittest discover -s tests
```

## Expected Output Files

Revision scripts write CSV/JSON/PDF/PNG files under `runs/revision_threshold_calibration/` and `runs/revision_qualitative_analysis/`. They do not save probability maps, intermediate features, or checkpoints.

## Citation

If you use this code, please cite:

```bibtex
@misc{etmnet,
  title = {ETMNet: An Edge-Guided Measurement Network for Real-Time Industrial Surface Defect Assessment},
  author = {Anonymous},
  note = {Citation information will be updated after publication}
}
```

No DOI, volume, issue, page range, or final publication year is claimed in this repository until those fields are officially available.

