import torch
import torch.nn as nn
import torch.nn.functional as F


def split_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 3 and isinstance(outputs[2], dict):
        logits, aux_outputs, extra = outputs
        return logits, tuple(aux_outputs), extra
    if isinstance(outputs, (tuple, list)):
        return outputs[0], tuple(outputs[1:]), {}
    return outputs, (), {}


class MeasurementOrientedDiceLoss(nn.Module):
    """Dice loss with per-sample defect-area weighting for measurement completeness."""

    def __init__(self, eps=1e-6, area_gamma=0.5, min_weight=0.5, max_weight=3.0):
        super().__init__()
        self.eps = float(eps)
        self.area_gamma = float(area_gamma)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def forward(self, logits, target):
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[:, 1]
        target = (target == 1).float()

        dims = (1, 2)
        intersection = torch.sum(probs * target, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)

        area = torch.sum(target, dim=dims)
        weights = torch.ones_like(area)
        positive = area > 0
        if torch.any(positive):
            mean_area = torch.mean(area[positive]).clamp_min(self.eps)
            weights[positive] = torch.pow(area[positive] / mean_area, self.area_gamma)
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        return torch.sum(dice_loss * weights) / torch.sum(weights).clamp_min(self.eps)


def make_boundary_target(masks, size, kernel_size=3):
    masks = masks.float().unsqueeze(1)
    padding = int(kernel_size) // 2
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - masks, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = torch.clamp(dilated - eroded, min=0.0, max=1.0)
    if boundary.shape[2:] != size:
        boundary = F.interpolate(boundary, size=size, mode="nearest")
    return (boundary > 0).float()


class ETMNetLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        measurement_dice_weight=1.0,
        aux_weight=0.4,
        boundary_weight=0.2,
        use_measurement_dice=True,
        area_gamma=0.5,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.measurement_dice_weight = float(measurement_dice_weight)
        self.aux_weight = float(aux_weight)
        self.boundary_weight = float(boundary_weight)
        self.use_measurement_dice = bool(use_measurement_dice)
        self.ce = nn.CrossEntropyLoss()
        self.dice = MeasurementOrientedDiceLoss(area_gamma=0.0)
        self.measurement_dice = MeasurementOrientedDiceLoss(area_gamma=area_gamma)
        self.boundary = nn.BCEWithLogitsLoss()

    def _segmentation_loss(self, logits, target):
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode="bilinear", align_corners=False)
        loss = logits.new_zeros(())
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, target.long())
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        if self.use_measurement_dice and self.measurement_dice_weight > 0:
            loss = loss + self.measurement_dice_weight * self.measurement_dice(logits, target)
        return loss

    def forward(self, outputs, target):
        logits, aux_outputs, extra = split_outputs(outputs)
        main_loss = self._segmentation_loss(logits, target)
        aux_loss = logits.new_zeros(())
        if aux_outputs:
            aux_values = [self._segmentation_loss(aux, target) for aux in aux_outputs]
            aux_loss = torch.stack(aux_values).mean()

        boundary_loss = logits.new_zeros(())
        boundary_logits = extra.get("boundary_logits")
        if self.boundary_weight > 0 and boundary_logits is not None:
            boundary_target = make_boundary_target(target, boundary_logits.shape[2:])
            boundary_loss = self.boundary(boundary_logits, boundary_target)

        return main_loss + self.aux_weight * aux_loss + self.boundary_weight * boundary_loss

