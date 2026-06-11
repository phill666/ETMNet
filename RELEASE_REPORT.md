# ETMNet Release Report

## Scope

This release directory contains only ETMNet-related source code, configs, documentation, and runnable command examples. Original datasets, checkpoints, experiment logs, large outputs, and unrelated model families were excluded.

## Core Files Included

- `models/etmnet.py`: ETMNet model with EdgeFlowConv and topology-aware context routing.
- `models/edgeflowconv.py`: EdgeFlowConv implementation.
- `models/pidnet_backbone.py`: real-time three-branch segmentation backbone.
- `losses/measurement_dice_loss.py`: measurement-oriented Dice weighting and ETMNet training loss.
- `metrics/segmentation_metrics.py`: mIoU, defect IoU, Dice/F1, Precision, Recall, and pixel accuracy.
- `metrics/measurement_metrics.py`: RAE, LE, BDE, CE, ConnE, Severity Acc, Severity MAE, and Spearman rho.
- `tools/train.py`, `tools/test.py`, `tools/threshold_calibration.py`, `tools/profile_model.py`, `tools/visualize_predictions.py`.
- `datasets/neu_seg.py`, `datasets/kolektorsdd2.py`, `datasets/magnetic_tile.py`.

## Excluded Content

- Raw dataset images, masks, archives, and extracted dataset folders.
- Trained weights and exported model files such as `.pth`, `.pt`, `.ckpt`, `.onnx`, and `.engine`.
- TensorBoard, wandb, run logs, checkpoints, outputs, result dumps, and temporary files.
- Large qualitative prediction outputs.
- Unrelated anomaly-detection, Crack500, CFRP, RealNet, DRAEM, EfficientAD, PatchCore, UNet, SegFormer, DDRNet, and older experiment modules.
- Local absolute paths and private machine-specific paths.

## Checks Performed

- `python -m compileall ETMNet_GitHub_Release`: passed.
- `ETMNet` import from `models/etmnet.py`: passed in an environment with PyTorch installed.
- `EdgeFlowConv` import from `models/edgeflowconv.py`: passed in an environment with PyTorch installed.
- `tools/profile_model.py` model-construction smoke check: passed on CPU with a 64x64 input.
- Publish hygiene scans found no bytecode caches, weight files, archives, logs, obvious credentials, or local absolute paths.

## Environment Note

The default shell Python used during cleanup did not have `torch` installed. Model import and profiler smoke checks were verified with a PyTorch-enabled environment. Install the packages in `requirements.txt` before training or testing.
