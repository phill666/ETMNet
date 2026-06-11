# ETMNet: An Edge-Guided Topology-Aware Measurement Network for Real-Time Industrial Surface Defect Assessment

This repository provides the official implementation of ETMNet for real-time industrial surface defect segmentation and measurement-oriented defect assessment.

## Method Overview

- ETMNet is built on a real-time three-branch segmentation framework.
- EdgeFlowConv is introduced into the boundary feature pathway.
- Measurement-oriented Dice weighting is used to improve defect-region completeness.
- Validation-based threshold calibration is used to obtain DiceOpt and MeasurementOpt thresholds.
- The code supports both segmentation metrics and measurement-oriented metrics.

## Repository Structure

```text
ETMNet_GitHub_Release/
|-- configs/                 # Dataset-specific YAML configs
|-- datasets/                # Dataset readers and preparation notes
|-- losses/                  # Measurement-oriented Dice weighting
|-- metrics/                 # Segmentation and measurement metrics
|-- models/                  # ETMNet, EdgeFlowConv, and real-time backbone
|-- tools/                   # Training, testing, calibration, profiling, visualization
|-- scripts/                 # Shell command examples
|-- assets/                  # Lightweight README assets
|-- README.md
|-- LICENSE
|-- requirements.txt
`-- .gitignore
```

## Installation

```bash
conda create -n etmnet python=3.10 -y
conda activate etmnet
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version if the default wheel is not suitable for your machine.

## Dataset Preparation

Datasets are not included in this repository. Please download NEU Seg, KolektorSDD2, and Magnetic Tile Defect from their official sources and arrange them as follows:

```text
data/
|-- NEU_Seg/
|   |-- images/
|   |-- masks/
|   `-- splits/
|-- KolektorSDD2/
|   |-- images/
|   |-- masks/
|   `-- splits/
`-- Magnetic_Tile_Defect/
    |-- images/
    |-- masks/
    `-- splits/
```

Each split file should contain one sample key per line. A key can be a stem such as `0001` or a relative key such as `defect_class/0001`; the readers match it against image and mask files with the same relative stem.

## Training

```bash
python tools/train.py --config configs/neu_seg_etmnet.yaml
python tools/train.py --config configs/kolektorsdd2_etmnet.yaml
python tools/train.py --config configs/magnetic_tile_etmnet.yaml
```

## Testing

```bash
python tools/test.py --config configs/neu_seg_etmnet.yaml --checkpoint checkpoints/etmnet_neu_seg.pth
python tools/test.py --config configs/kolektorsdd2_etmnet.yaml --checkpoint checkpoints/etmnet_ksdd2.pth
python tools/test.py --config configs/magnetic_tile_etmnet.yaml --checkpoint checkpoints/etmnet_mtd.pth
```

## Threshold Calibration

DiceOpt focuses on segmentation overlap performance. MeasurementOpt focuses on measurement-oriented reliability.

```bash
python tools/threshold_calibration.py --config configs/magnetic_tile_etmnet.yaml --checkpoint checkpoints/etmnet_mtd.pth --mode both
```

## Metric Evaluation

Segmentation metrics:

```bash
python tools/evaluate_segmentation.py --pred-dir outputs/mtd/pred --gt-dir data/Magnetic_Tile_Defect/masks
```

Measurement-oriented metrics:

```bash
python tools/evaluate_measurement.py --pred-dir outputs/mtd/pred --gt-dir data/Magnetic_Tile_Defect/masks
```

The segmentation evaluator reports mIoU, defect IoU, Dice/F1, Precision, and Recall. The measurement evaluator reports RAE, LE, BDE, CE, ConnE, Severity Acc, Severity MAE, and Spearman rho.

## Complexity and Speed

```bash
python tools/profile_model.py --config configs/magnetic_tile_etmnet.yaml --input-size 512 512
```

The profiler reports Params, GFLOPs, FPS, and latency. GFLOPs use `thop` when available and a convolution/linear fallback otherwise.

## Visualization

```bash
python tools/visualize_predictions.py --config configs/magnetic_tile_etmnet.yaml --checkpoint checkpoints/etmnet_mtd.pth --output-dir outputs/mtd/vis
```

## Pretrained Weights

The current GitHub version does not upload trained weights.

Trained weights will be made available upon reasonable request.

## Code Availability

This code is provided for academic research and reproducibility.

## Citation

If you use this code, please cite our paper:

ETMNet: An Edge-Guided Topology-Aware Measurement Network for Real-Time Industrial Surface Defect Assessment.

Citation information will be updated after publication.

