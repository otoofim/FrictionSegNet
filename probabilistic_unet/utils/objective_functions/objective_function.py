import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss.

    Computes MSE loss only on non-zero target values, useful for scenarios
    where zero values represent missing or invalid data that should be ignored.

    Args:
        eps (float): Small epsilon value to prevent division by zero. Default: 1e-15
    """

    def __init__(self, eps: float = 1e-15):
        super(MaskedMSELoss, self).__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            output (torch.Tensor): Predicted values
            target (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Create mask for non-zero targets
        mask = (target.detach() != 0).float()

        # Compute squared error
        squared_error = (target - output) ** 2

        # Apply mask and compute mean
        masked_loss = squared_error * mask
        loss = masked_loss.sum() / (mask.sum() + self.eps)

        return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing.

    Wrapper around PyTorch's CrossEntropyLoss that computes spatial mean
    over height and width dimensions after computing the loss.

    Args:
        label_smoothing (float): Label smoothing factor. Default: 0.0
        reduction (str): Reduction method for pixel-wise losses. Default: "none"
    """

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "none"):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=label_smoothing
        )

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss with spatial averaging.

        Args:
            output (torch.Tensor): Predicted logits of shape (N, C, H, W)
            target (torch.Tensor): Ground truth labels of shape (N, H, W)

        Returns:
            torch.Tensor: Loss value averaged over spatial dimensions (N,)
        """
        # Compute pixel-wise cross-entropy and average over spatial dimensions
        loss = self.criterion(output, target)
        return torch.mean(loss, dim=(1, 2))


def compute_l2_regularization(model: nn.Module) -> torch.Tensor:
    """
    Compute L2 regularization term for model parameters.

    Sums the L2 norms of all model parameters, which can be added to the loss
    function to penalize large weights and prevent overfitting.

    Args:
        model (nn.Module): The model whose parameters will be regularized

    Returns:
        torch.Tensor: Scalar tensor containing the sum of L2 norms

    Example:
        >>> model = MyModel()
        >>> l2_reg = compute_l2_regularization(model)
        >>> loss = base_loss + lambda_reg * l2_reg
    """
    l2_norm = torch.tensor(0.0, device=next(model.parameters()).device)

    for param in model.parameters():
        if param.requires_grad:
            l2_norm = l2_norm + param.norm(2)

    return l2_norm
