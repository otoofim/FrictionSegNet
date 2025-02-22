import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy as np
import wandb
import logging
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from sklearn import metrics
import torch.nn as nn

sys.path.insert(1, "./architecture")
sys.path.insert(2, "./dataLoaders")

from probabilistic_unet.utils.config_loader.config_manager import ConfigManager
from probabilistic_unet.dataloader.CityscapesLoader import CityscapesLoader
from probabilistic_unet.model.ProUNet import ProUNet


@dataclass
class TrainingMetrics:
    """Container for tracking training and validation metrics"""

    training: Dict[str, List]
    validation: Dict[str, List]

    @classmethod
    def initialize(cls):
        base_metrics = {
            "Total loss": [],
            "KL loss": [],
            "Reconstruction loss": [],
            "mIoU": [],
            "ECE": [],
            "MCE": [],
            "RMSCE": [],
            "NLL": [],
            "Brier": [],
            "Regression loss": [],
            "Confusion Matrix": [],
        }

        training = base_metrics.copy()
        training.update(
            {
                "lambda_fri_coefficient": [],
                "lambda_seg_coefficient": [],
            }
        )

        return cls(training=training, validation=base_metrics.copy())


class Trainer:
    def __init__(self, configs: ConfigManager):
        self.configs = configs
        self.setup_logging()
        self.setup_paths()
        self.metrics = TrainingMetrics.initialize()
        self.device = self._setup_device()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{self.configs.project_name}.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_paths(self):
        """Setup training paths and directories"""
        self.checkpoint_dir = Path("checkpoints")
        self.model_path = (
            self.checkpoint_dir / f"{self.configs.project_name}_{self.configs.run_name}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict:
        """Execute one training epoch"""
        tr_metrics = TrainingMetrics.initialize().training

        with tqdm(train_loader, unit="batch", leave=False) as batbar:
            for i, batch in enumerate(batbar):
                batbar.set_description(f"Batch {i + 1}")

                # Zero gradients
                optimizer.zero_grad()
                model.train()

                # Move data to device
                batch_img = batch["image"].to(self.device)
                batch_label = batch["label"].to(self.device)
                fri_label = batch["FriLabel"].to(self.device)

                # Forward pass
                seg, prior_dists, posterior_dists, fri_pred = model(
                    batch_img, batch_label, fri_label
                )

                # Calculate losses
                (
                    loss,
                    kl_mean,
                    kl_losses,
                    rec_loss,
                    miou,
                    ious,
                    l1_loss,
                    l2_loss,
                    l3_loss,
                    reg_loss,
                    cm,
                ) = model.loss(
                    batch_label, seg, prior_dists, posterior_dists, fri_label, fri_pred
                )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update metrics
                self._update_training_metrics(
                    tr_metrics,
                    model,
                    batch,
                    seg,
                    loss,
                    kl_mean,
                    rec_loss,
                    miou,
                    l1_loss,
                    l2_loss,
                    l3_loss,
                    reg_loss,
                    cm,
                    kl_losses,
                    ious,
                )

                # Save intermediate checkpoints
                if i % 1000 == 0:
                    self.save_checkpoint(
                        epoch, model, optimizer, tr_metrics, name="iterative"
                    )

        # Log training visualizations
        self._log_training_visualizations(batch, seg, epoch, tr_metrics)

        return tr_metrics

    def _validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, epoch: int
    ) -> Dict:
        """Execute one validation epoch"""
        val_metrics = TrainingMetrics.initialize().validation

        with torch.no_grad(), tqdm(val_loader, unit="batch", leave=False) as valbar:
            for i, batch in enumerate(valbar):
                valbar.set_description(f"Val_batch {i + 1}")
                model.eval()

                # Move data to device
                batch_img = batch["image"].to(self.device)
                batch_label = batch["label"].to(self.device)
                fri_label = batch["FriLabel"].to(self.device)

                # Get model predictions
                samples, priors, posterior_dists, fri_preds = model.evaluation(
                    batch_img, batch_label, fri_label
                )

                # Calculate validation metrics for each sample
                self._calculate_validation_metrics(
                    val_metrics,
                    model,
                    batch,
                    samples,
                    priors,
                    posterior_dists,
                    batch_label,
                    fri_label,
                    fri_preds,
                )

        # Log validation visualizations
        self._log_validation_visualizations(batch, samples, epoch, val_metrics)

        return val_metrics

    def _update_training_metrics(
        self, metrics: Dict, model: nn.Module, batch: Dict, seg: torch.Tensor, *args
    ):
        """Update training metrics dictionary"""
        (
            loss,
            kl_mean,
            rec_loss,
            miou,
            l1_loss,
            l2_loss,
            l3_loss,
            reg_loss,
            cm,
            kl_losses,
            ious,
        ) = args

        metrics["Total loss"].append(loss.detach().cpu().item())
        metrics["KL loss"].append(kl_mean.detach().cpu().item())
        metrics["Reconstruction loss"].append(rec_loss.detach().cpu().item())
        metrics["mIoU"].append(torch.mean(miou).detach().cpu().item())
        metrics["ECE"].append(l1_loss.detach().cpu().item())
        metrics["MCE"].append(l2_loss.detach().cpu().item())
        metrics["RMSCE"].append(l3_loss.detach().cpu().item())

        if self.configs.GECO["enable"]:
            metrics["lambda_fri_coefficient"].append(
                model.geco.lambda_f.detach().cpu().item()
            )
            metrics["lambda_seg_coefficient"].append(
                model.geco.lambda_s.detach().cpu().item()
            )

        # Add KL losses and IoUs for each layer
        self._update_layer_metrics(metrics, kl_losses, "KL loss layer")
        self._update_layer_metrics(metrics, ious, "iou of")

    def _calculate_validation_metrics(
        self, metrics: Dict, model: nn.Module, batch: Dict, samples: torch.Tensor, *args
    ) -> None:
        """Calculate and update validation metrics"""
        priors, posterior_dists, batch_label, fri_label, fri_preds = args

        for i, sample in enumerate(samples):
            (
                loss,
                kl_mean,
                kl_losses,
                rec_loss,
                miou,
                ious,
                l1_loss,
                l2_loss,
                l3_loss,
                reg_loss,
                cm,
            ) = model.loss(
                batch_label, sample, priors, posterior_dists, fri_label, fri_preds[i]
            )

            self._update_validation_sample_metrics(
                metrics,
                loss,
                kl_mean,
                rec_loss,
                miou,
                l1_loss,
                l2_loss,
                l3_loss,
                reg_loss,
                cm,
                kl_losses,
                ious,
                sample,
                batch_label,
            )

    def train(self):
        """Main training loop"""
        self.initialize_wandb()
        train_loader, val_loader = self.setup_datasets()
        model = self.setup_model(train_loader.dataset.datasets[0].get_num_classes())
        optimizer, scheduler = self.setup_optimizer(model)

        start_epoch = self._get_start_epoch()
        best_val_miou = -1

        wandb.watch(model, log_freq=100)

        for epoch in tqdm(
            range(start_epoch, self.configs.epochs),
            initial=start_epoch,
            total=self.configs.epochs,
        ):
            try:
                # Training phase
                tr_metrics = self._train_epoch(model, train_loader, optimizer, epoch)

                # Validation phase
                if (epoch + 1) % self.configs.val_frequency == 0:
                    val_metrics = self._validate_epoch(model, val_loader, epoch)

                    # Update learning rate scheduler
                    val_miou = np.mean(val_metrics["mIoU"])
                    scheduler.step(val_miou)

                    # Save best model
                    if val_miou > best_val_miou:
                        best_val_miou = val_miou
                        self.save_checkpoint(
                            epoch,
                            model,
                            optimizer,
                            {"train": tr_metrics, "val": val_metrics},
                            name="best",
                        )

                # Log metrics
                self._log_metrics(epoch, tr_metrics, val_metrics)

            except Exception as e:
                self.logger.error(f"Error during epoch {epoch}: {str(e)}")
                raise

        self.logger.info("Training completed")
        wandb.finish()


# Additional helper methods would go here (visualization, metric logging, etc.)
# Implementation details omitted for brevity

if __name__ == "__main__":
    configs = ConfigManager()
    trainer = Trainer(configs)
    trainer.train()
