# -*- coding: utf-8 -*-
"""


@author: Fu Haitao
"""


import torch
import torch.nn.functional as F


def confidence_penalty(predictions, penalty_weight=0.0):
    """
    Confidence penalty discouraging extreme output probabilities near 0 or 1.
    predictions: probabilities after sigmoid (0 to 1)
    penalty_weight: penalty coefficient; recommended range 0.001 to 0.01
    """
    if penalty_weight == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Confidence increases with distance from 0.5 (range: 0 to 1).
    conf = torch.abs(predictions - 0.5) * 2
    # Penalty is -log(1 - conf) and diverges as conf approaches 1.
    # Add 1e-8 to avoid log(0).
    penalty = -torch.log(1 - conf + 1e-8)
    return penalty_weight * penalty.mean()


# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def focal_loss(predictions, targets, gamma=2.0, alpha=0.25, reduction='none'):
    """
    Focal loss with per-sample computation.
    predictions: probabilities (0 to 1) or logits
    targets: binary labels
    reduction: 'none' returns per-sample losses; 'mean' returns their mean
    """
    # Use BCE for probabilities and BCE-with-logits otherwise.
    if predictions.min() >= 0 and predictions.max() <= 1:
        ce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
    else:
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')

    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    return loss


def lossF(lossType, predictions, targets, gamma=2.0, alpha=0.25, sample_weight=None):
    """
    Unified loss function supporting per-sample weights.

    Args:
        predictions: model predictions (batch_size,)
        targets: ground-truth labels (batch_size,)
        sample_weight: per-sample weights (batch_size,), or None by default
    """
    # Compute per-sample losses with reduction='none'.
    if lossType == 'cross_entropy':
        # Infer whether the inputs are logits.
        if (predictions.min() < 0) or (predictions.max() > 1):
            losses = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        else:
            losses = F.binary_cross_entropy(predictions, targets, reduction='none')

    elif lossType == 'MF_all':
        losses = torch.pow((predictions - targets), 2)  # Per-sample squared error.

    elif lossType == 'MSE':
        losses = F.mse_loss(predictions, targets, reduction='none')

    elif lossType == 'focal_loss':
        losses = focal_loss(predictions, targets, gamma=gamma, alpha=alpha, reduction='none')

    else:
        raise ValueError(f"Unsupported loss type: {lossType}")

    # ========== Apply per-sample weights ==========
    if sample_weight is not None:
        losses = losses * sample_weight
    # =================================

    return losses.mean()
