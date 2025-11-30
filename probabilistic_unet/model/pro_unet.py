from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torchmetrics.functional.classification import multiclass_calibration_error

from probabilistic_unet.model.posterior import Posterior
from probabilistic_unet.model.prior import Prior
from probabilistic_unet.utils.confusion_matrix.confusion_matrix import (
    BatchImageConfusionMatrix,
    multiclass_iou,
)
from probabilistic_unet.utils.objective_functions.objective_function import (
    CrossEntropyLoss,
)


def init_weights(m: nn.Module) -> None:
    """Initialize weights for convolutional layers.

    Args:
        m: Module to initialize
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=0.001)


class ProUNet(nn.Module):
    """
    Probabilistic U-Net for semantic segmentation with uncertainty quantification.

    This model combines a prior and posterior network to learn a distribution over
    segmentation masks, enabling uncertainty estimation through sampling.

    Args:
        num_classes: Number of segmentation classes
        device: Device to run the model on (kept for compatibility, but not actively used)
        latent_var_size: Dimensionality of the latent space (default: 6)
        beta: Weight for KL divergence in ELBO loss (default: 5.0)
        use_posterior: Whether to use posterior network for training (default: True)
        num_samples: Number of samples to generate during inference (default: 16)
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        latent_var_size: int = 6,
        beta: float = 5.0,
        use_posterior: bool = True,
        num_samples: int = 16,
    ):
        super().__init__()
        # Model configuration
        self.latent_var_size = latent_var_size
        self.beta = beta
        self.use_posterior = use_posterior
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.device = device  # Kept for backward compatibility

        # Architecture
        self.prior = Prior(
            num_samples=self.num_samples,
            num_classes=self.num_classes,
            latent_var_size=self.latent_var_size,
            input_dim=3,
            base_channels=128,
            num_res_layers=2,
            activation=nn.ReLU,
        ).apply(init_weights)

        if self.use_posterior:
            self.posterior = Posterior(
                num_samples=self.num_samples,
                num_classes=self.num_classes,
                latent_var_size=self.latent_var_size,
                input_dim=4,  # 3 RGB channels + 1 class index channel
                base_channels=128,
                num_res_layers=2,
                activation=nn.ReLU,
            ).apply(init_weights)
        else:
            self.posterior = None

        # Loss function
        self.criterion = CrossEntropyLoss(label_smoothing=0.4)

    def forward(
        self, input_img: torch.Tensor, segmasks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Forward pass through the model.

        Args:
            input_img: Input image tensor [B, 3, H, W]
            segmasks: Ground truth segmentation class indices [B, 1, H, W]

        Returns:
            Tuple of (segmentation_output, prior_distributions, posterior_distributions)
        """
        if self.posterior is None or segmasks is None:
            raise ValueError("Posterior network and segmasks required for forward pass")

        posterior_dists = self.posterior(torch.cat((input_img, segmasks), dim=1))
        seg, prior_dists = self.prior(input_img, post_dist=posterior_dists)

        return seg, prior_dists, posterior_dists

    def inference(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Generate samples from the prior network during inference.

        Args:
            input_features: Input image tensor [B, 3, H, W]

        Returns:
            Tuple of (samples, distributions) where samples is [B*num_samples, num_classes, H, W]
        """
        return self.prior.inference(input_features)

    def latent_visualize(
        self,
        input_features: torch.Tensor,
        sample_latent1: Optional[torch.Tensor] = None,
        sample_latent2: Optional[torch.Tensor] = None,
        sample_latent3: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate samples with custom latent variables for visualization.

        Args:
            input_features: Input image tensor [B, 3, H, W]
            sample_latent1: Custom latent sample for level 1
            sample_latent2: Custom latent sample for level 2
            sample_latent3: Custom latent sample for level 3

        Returns:
            Tuple of (samples, distributions) with custom latent samples
        """
        return self.prior.latentVisualize(
            input_features,
            sample_latent1=sample_latent1,
            sample_latent2=sample_latent2,
            sample_latent3=sample_latent3,
        )

    def evaluation(
        self, input_features: torch.Tensor, segmasks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Evaluation mode: generate samples and compute distributions.

        Args:
            input_features: Input image tensor [B, 3, H, W]
            segmasks: Ground truth segmentation class indices [B, 1, H, W]

        Returns:
            Tuple of (samples, prior_distributions, posterior_distributions)
        """
        if self.posterior is None:
            raise ValueError("Posterior network required for evaluation")

        with torch.no_grad():
            samples, priors = self.prior.inference(input_features)
            posterior_dists = self.posterior.inference(
                torch.cat((input_features, segmasks), dim=1)
            )
            return samples, priors, posterior_dists

    def reconstruction_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss between predictions and targets.

        Args:
            predictions: Model predictions [B, num_classes, H, W]
            targets: Ground truth segmentation class indices [B, 1, H, W]

        Returns:
            Reconstruction loss value
        """
        # Squeeze targets from [B, 1, H, W] to [B, H, W] for CrossEntropyLoss
        targets_squeezed = torch.squeeze(targets, dim=1).long()
        return self.criterion(output=predictions, target=targets_squeezed)

    def kl_loss(self, priors: Dict, posteriors: Dict) -> Dict[int, torch.Tensor]:
        """Compute KL divergence between posterior and prior distributions.

        Args:
            priors: Dictionary of prior distributions at each level
            posteriors: Dictionary of posterior distributions at each level

        Returns:
            Dictionary mapping level to KL divergence tensor
        """
        kl_losses = {}
        for level, (posterior, prior) in enumerate(
            zip(posteriors.items(), priors.items())
        ):
            kl_losses[level] = torch.mean(
                kl_divergence(posterior[1], prior[1]), dim=(1, 2)
            )
        return kl_losses

    def elbo_loss(
        self,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        priors: Dict,
        posteriors: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:
        """Compute Evidence Lower Bound (ELBO) loss.

        ELBO = Reconstruction Loss + β * KL Divergence

        Args:
            targets: Ground truth segmentation class indices [B, 1, H, W]
            predictions: Model predictions [B, num_classes, H, W]
            priors: Dictionary of prior distributions
            posteriors: Dictionary of posterior distributions

        Returns:
            Tuple of (total_loss, kl_mean, kl_losses_dict, reconstruction_loss)
        """
        rec_loss = torch.mean(self.reconstruction_loss(predictions, targets))

        kl_losses = self.kl_loss(priors, posteriors)
        # Stack and sum KL losses across all levels
        kl_mean = torch.mean(torch.sum(torch.stack(list(kl_losses.values())), dim=0))

        # ELBO: reconstruction + weighted KL
        total_loss = rec_loss + self.beta * kl_mean

        return total_loss, kl_mean, kl_losses, rec_loss

    def compute_stats(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute various evaluation statistics.

        Args:
            predictions: Model predictions [B, num_classes, H, W]
            labels: Ground truth segmentation class indices [B, 1, H, W]

        Returns:
            Tuple of (mean_iou, class_ious, l1_calibration, l2_calibration,
                     max_calibration, confusion_matrix)
        """
        # Generate class indices dynamically based on num_classes
        class_indices = list(range(self.num_classes))

        miou, ious = multiclass_iou(predictions, labels, class_indices)
        confusion_matrix = BatchImageConfusionMatrix(predictions, labels, class_indices)

        # Convert class indices from [B, 1, H, W] to [B, H, W]
        target_indices = torch.squeeze(labels, dim=1).long()

        l1_calibration = multiclass_calibration_error(
            preds=predictions,
            target=target_indices,
            num_classes=self.num_classes,
            n_bins=10,
            norm="l1",
        )
        l2_calibration = multiclass_calibration_error(
            preds=predictions,
            target=target_indices,
            num_classes=self.num_classes,
            n_bins=10,
            norm="l2",
        )
        max_calibration = multiclass_calibration_error(
            preds=predictions,
            target=target_indices,
            num_classes=self.num_classes,
            n_bins=10,
            norm="max",
        )

        return (
            miou,
            ious,
            l1_calibration,
            l2_calibration,
            max_calibration,
            confusion_matrix,
        )

    def compute_loss(
        self,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        priors: Dict,
        posteriors: Dict,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute total loss and evaluation metrics.

        Args:
            targets: Ground truth segmentation class indices [B, 1, H, W]
            predictions: Model predictions [B, num_classes, H, W]
            priors: Dictionary of prior distributions
            posteriors: Dictionary of posterior distributions

        Returns:
            Tuple of (total_loss, kl_mean, kl_losses_dict, reconstruction_loss,
                     mean_iou, class_ious, l1_calibration, l2_calibration,
                     max_calibration, confusion_matrix)
        """
        total_loss, kl_mean, kl_losses, rec_loss = self.elbo_loss(
            targets, predictions, priors, posteriors
        )

        miou, ious, l1_cal, l2_cal, max_cal, cm = self.compute_stats(
            predictions, targets
        )

        return (
            total_loss,
            kl_mean,
            kl_losses,
            rec_loss,
            miou,
            ious,
            l1_cal,
            l2_cal,
            max_cal,
            cm,
        )
