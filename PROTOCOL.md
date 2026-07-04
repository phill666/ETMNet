# ETMNet Revision Evaluation Protocol

Protocol version: `revision_2026_07`

## Segmentation Evaluation

Models are evaluated in `eval()` mode with fixed BatchNorm statistics. The final segmentation output is used. Auxiliary outputs such as PIDNet boundary or detail heads are not used for metrics.

Two-channel logits are converted with softmax and the defect probability is the foreground channel. Single-channel outputs may use sigmoid only when the model was trained with BCE. Thresholds are applied to probabilities, not logits.

Prediction probabilities are resized to the original image size before thresholding. The public revision scripts use bilinear restoration with `align_corners=True` for logits/probability maps. Binary masks and GT masks use nearest-neighbor handling only. Segmentation metrics accumulate a dataset-level confusion matrix and include normal images.

## Geometry Evaluation

RAE, LE, BDE, CE, and ConnE are averaged only over GT-positive samples. GT-empty samples are counted for diagnostics but excluded from geometry-error means.

- `RAE`: relative area error.
- `LE`: relative skeleton graph length error using 8-neighbor graph edges.
- `BDE`: bidirectional distance between one-pixel binary boundaries.
- `CE`: centroid Euclidean distance in original-image pixels.
- `ConnE`: absolute 8-neighbor connected-component count difference.

For GT-positive empty predictions, BDE and CE use the image diagonal penalty.

Note: the historical revision evaluation script used for internal experiments computed LE using skeleton pixel count and derived severity normalization within the evaluated split. The public protocol documents the finalized repository implementation with training-set severity metadata. Historical result files are not recomputed or overwritten.

## Severity Protocol

Severity metadata is generated once per dataset from training GT masks only. Area, skeleton length, and boundary irregularity are min-max normalized using training-set statistics and clipped to `[0, 1]`. The continuous severity score is the mean of the three normalized values. Low, moderate, and high severity classes use training-set 33.3% and 66.7% quantiles.

Severity Accuracy compares predicted and GT severity classes. Severity MAE is the absolute error between continuous severity scores. Spearman rho is computed from continuous severity scores.

## Threshold Calibration

Candidate thresholds are fixed to `0.05, 0.10, ..., 0.95`.

DiceOpt maximizes validation Dice/F1. MeasurementOpt minimizes validation `normalized_RAE + normalized_BDE + normalized_CE`, where each metric is min-max normalized over the candidate thresholds for that model. Constant metrics normalize to zero and never produce NaN. Ties choose the threshold closest to `0.50`; remaining ties choose the smaller threshold.

The test set is used only after the validation threshold is fixed.

## Dataset Split Rules

Training, validation, and test splits are fixed. The test split is not used for checkpoint selection, threshold selection, severity metadata, or metric normalization.

## Model Output Adapters

Each model adapter must record model class, checkpoint loader, final output selector, probability conversion, preprocessing, output resize protocol, and foreground class index. Missing or mismatched critical checkpoint parameters must stop evaluation.

## Revision History

- `revision_2026_07`: public repository protocol aligned with the revised edge-guided measurement framing, validation-based threshold calibration, and training-GT severity metadata.
