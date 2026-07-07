import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification tasks.
    Focuses training on hard, misclassified examples and reduces loss contribution of easy examples.
    Supports class-specific positive weighting.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor = None, reduction: str = "mean"):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        inputs: raw logits of shape (batch_size, num_classes)
        targets: binary targets of shape (batch_size, num_classes)
        """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy loss (with pos_weight if provided)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction="none")

        # Compute the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weighting factor: (1 - p_t) ^ gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Combine
        loss = focal_weight * bce_loss

        # Apply alpha balance factor
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
