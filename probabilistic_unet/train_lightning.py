import sys
from dataclasses import asdict
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

# TorchMetrics for efficient metric computation
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassCalibrationError,
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
)

from probabilistic_unet.dataloader.cityscapes_loader import (
    CITYSCAPES_CLASSES,
    NUM_CITYSCAPES_CLASSES,
    CityscapesDatasetConfig,
    create_cityscapes_dataloaders,
)
from probabilistic_unet.model.pro_unet import ProUNet

# Local imports
from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig
from probabilistic_unet.dataloader.lightning_dataloader import (
    FrictionSegNetDataModule,
)

# Configure loguru for beautiful logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)
logger.add(
    "logs/training_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG"
)


class FrictionSegNetLightning(pl.LightningModule):
    """
    PyTorch Lightning Module for FrictionSegNet.

    This encapsulates the training logic, metrics, and optimization
    in a clean, reusable way following Lightning best practices.
    """

    def __init__(
        self,
        num_classes: int,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["training_config"])

        self.config = training_config
        self.num_classes = num_classes

        # Initialize model with new clean API
        self.model = ProUNet(
            num_classes=num_classes,
            device=self.device,
            latent_var_size=training_config.latent_dim,
            beta=training_config.beta,
            use_posterior=training_config.use_posterior,
            num_samples=training_config.num_samples,
        )

        # Setup metrics using TorchMetrics for efficiency
        self._setup_metrics()

        # Track best metrics
        self.best_val_miou = 0.0

        logger.info(f"Initialized FrictionSegNet with {num_classes} classes")
        logger.info(f"Latent dimension: {training_config.latent_dim}")
        logger.info(f"Beta (KL weight): {training_config.beta}")
        logger.info(f"Number of samples: {training_config.num_samples}")

    def _setup_metrics(self):
        """Setup metrics for training and validation using TorchMetrics."""

        # Loss metrics
        self.train_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.train_rec_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_kl_loss = MeanMetric()
        self.val_rec_loss = MeanMetric()

        # Segmentation metrics
        self.train_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes, average="macro", ignore_index=None
        )

        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes, average="macro", ignore_index=None
        )

        # Per-class IoU for detailed analysis
        self.val_iou_per_class = MulticlassJaccardIndex(
            num_classes=self.num_classes, average="none", ignore_index=None
        )

        # Calibration metrics (uncertainty estimation quality)
        self.val_ece = MulticlassCalibrationError(
            num_classes=self.num_classes, n_bins=10, norm="l1"
        )

        self.val_mce = MulticlassCalibrationError(
            num_classes=self.num_classes, n_bins=10, norm="max"
        )

        # Confusion matrix for detailed error analysis
        self.val_confusion = MulticlassConfusionMatrix(num_classes=self.num_classes)

        # Additional quality metrics
        self.train_nll = MeanMetric()
        self.train_brier = MeanMetric()
        self.val_nll = MeanMetric()
        self.val_brier = MeanMetric()

    def forward(self, x: torch.Tensor, segmasks: Optional[torch.Tensor] = None):
        """Forward pass through the model."""
        return self.model(x, segmasks)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step - called for each batch during training."""
        images = batch["image"]
        labels = batch["label"]

        # Forward pass
        predictions, priors, posteriors = self(images, labels)

        # Compute loss using model's clean API
        (
            total_loss,
            kl_mean,
            kl_losses,
            rec_loss,
            miou,
            ious,
            ece,
            mce,
            rmsce,
            confusion_matrix,
        ) = self.model.compute_loss(labels, predictions, priors, posteriors)

        # Update metrics
        self.train_loss(total_loss)
        self.train_kl_loss(kl_mean)
        self.train_rec_loss(rec_loss)

        # Convert predictions and labels for metric computation
        pred_classes = torch.argmax(predictions, dim=1)
        label_classes = torch.argmax(labels, dim=1)
        self.train_miou(pred_classes, label_classes)

        # Additional quality metrics
        with torch.no_grad():
            nll = F.nll_loss(
                F.log_softmax(predictions, dim=1), label_classes, reduction="mean"
            )
            brier = torch.mean((predictions - labels) ** 2)

            self.train_nll(nll)
            self.train_brier(brier)

        # Log metrics
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/kl_loss", self.train_kl_loss, on_step=True, on_epoch=True)
        self.log("train/rec_loss", self.train_rec_loss, on_step=True, on_epoch=True)
        self.log(
            "train/mIoU", self.train_miou, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/nll", self.train_nll, on_step=False, on_epoch=True)
        self.log("train/brier", self.train_brier, on_step=False, on_epoch=True)

        # Log layer-specific KL losses
        for layer_id, kl_loss in kl_losses.items():
            self.log(
                f"train/kl_layer_{layer_id}",
                torch.mean(kl_loss),
                on_step=False,
                on_epoch=True,
            )

        # Log per-class IoU
        for class_id, iou in ious.items():
            class_name = CITYSCAPES_CLASSES.get(class_id, f"class_{class_id}")
            self.log(
                f"train/iou_{class_name}", torch.mean(iou), on_step=False, on_epoch=True
            )

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step - called for each batch during validation."""
        images = batch["image"]
        labels = batch["label"]

        # Generate multiple samples for better uncertainty estimation
        samples, priors, posteriors = self.model.evaluation(images, labels)

        # Compute metrics across all samples
        batch_losses = []
        batch_kl = []
        batch_rec = []

        for sample_idx, sample in enumerate(samples):
            (
                loss,
                kl_mean,
                kl_losses,
                rec_loss,
                miou,
                ious,
                ece,
                mce,
                rmsce,
                confusion_matrix,
            ) = self.model.compute_loss(labels, sample, priors, posteriors)

            batch_losses.append(loss)
            batch_kl.append(kl_mean)
            batch_rec.append(rec_loss)

        # Average across samples
        avg_loss = torch.mean(torch.stack(batch_losses))
        avg_kl = torch.mean(torch.stack(batch_kl))
        avg_rec = torch.mean(torch.stack(batch_rec))

        # Use mean prediction for metrics
        mean_prediction = torch.mean(samples, dim=0)

        # Update metrics
        self.val_loss(avg_loss)
        self.val_kl_loss(avg_kl)
        self.val_rec_loss(avg_rec)

        # Convert to class indices
        pred_classes = torch.argmax(mean_prediction, dim=1)
        label_classes = torch.argmax(labels, dim=1)

        self.val_miou(pred_classes, label_classes)
        self.val_iou_per_class(pred_classes, label_classes)
        self.val_ece(mean_prediction, label_classes)
        self.val_mce(mean_prediction, label_classes)
        self.val_confusion(pred_classes, label_classes)

        # Additional metrics
        nll = F.nll_loss(
            F.log_softmax(mean_prediction, dim=1), label_classes, reduction="mean"
        )
        brier = torch.mean((mean_prediction - labels) ** 2)

        self.val_nll(nll)
        self.val_brier(brier)

        # Log sample predictions for visualization (first batch only)
        if batch_idx == 0:
            self._log_predictions(images, labels, mean_prediction, samples)

        return avg_loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to log aggregated metrics."""

        # Log aggregated metrics
        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/kl_loss", self.val_kl_loss)
        self.log("val/rec_loss", self.val_rec_loss)
        self.log("val/mIoU", self.val_miou, prog_bar=True)
        self.log("val/ece", self.val_ece, prog_bar=True)
        self.log("val/mce", self.val_mce)
        self.log("val/nll", self.val_nll)
        self.log("val/brier", self.val_brier)

        # Log per-class IoU
        per_class_iou = self.val_iou_per_class.compute()
        for class_id, iou_val in enumerate(per_class_iou):
            class_name = CITYSCAPES_CLASSES.get(class_id, f"class_{class_id}")
            self.log(f"val/iou_{class_name}", iou_val)

        # Log confusion matrix as image
        cm = self.val_confusion.compute()
        if self.logger and hasattr(self.logger, "experiment"):
            self._log_confusion_matrix(cm)

        # Track best validation mIoU
        current_miou = self.val_miou.compute()
        if current_miou > self.best_val_miou:
            self.best_val_miou = current_miou
            logger.success(f"New best validation mIoU: {current_miou:.4f}")

        # Log to console
        logger.info(
            f"Validation - Loss: {self.val_loss.compute():.4f}, "
            f"mIoU: {current_miou:.4f}, "
            f"ECE: {self.val_ece.compute():.4f}"
        )

    def _log_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        samples: torch.Tensor,
    ):
        """Log sample predictions to WandB for visualization."""
        if not self.logger or not hasattr(self.logger, "experiment"):
            return

        # Take first 4 images from batch
        num_vis = min(4, images.size(0))

        wandb_images = []
        for i in range(num_vis):
            # Convert to numpy for visualization
            img = images[i].cpu()
            label = torch.argmax(labels[i], dim=0).cpu()
            pred = torch.argmax(predictions[i], dim=0).cpu()

            # Log as WandB image with masks
            wandb_images.append(
                wandb.Image(
                    img,
                    masks={
                        "ground_truth": {"mask_data": label.numpy()},
                        "prediction": {"mask_data": pred.numpy()},
                    },
                )
            )

        self.logger.experiment.log(
            {"val/predictions": wandb_images, "global_step": self.global_step}
        )

    def _log_confusion_matrix(self, cm: torch.Tensor):
        """Log confusion matrix as image to WandB."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Normalize confusion matrix
        cm_normalized = cm.float() / cm.sum(dim=1, keepdim=True)
        cm_normalized = torch.nan_to_num(cm_normalized, 0.0)

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(
            cm_normalized.cpu().numpy(),
            annot=False,
            fmt=".2f",
            cmap="Blues",
            xticklabels=list(CITYSCAPES_CLASSES.values()),
            yticklabels=list(CITYSCAPES_CLASSES.values()),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Normalized)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Log to WandB
        self.logger.experiment.log(
            {"val/confusion_matrix": wandb.Image(fig), "global_step": self.global_step}
        )
        plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""

        # Select optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Select scheduler
        if self.config.lr_scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.config.lr_scheduler.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.monitor_mode,
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.monitor_metric,
                    "interval": "epoch",
                },
            }

        elif self.config.lr_scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        else:  # no scheduler
            return optimizer


def train_frictionsegnet(
    dataset_config: CityscapesDatasetConfig,
    training_config: TrainingConfig,
):
    """
    Main training function using PyTorch Lightning.

    This provides a clean, modern interface for training with all
    the benefits of Lightning: automatic GPU usage, mixed precision,
    gradient clipping, checkpointing, logging, etc.
    """

    logger.info("=" * 80)
    logger.info("FrictionSegNet Training with PyTorch Lightning")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    pl.seed_everything(training_config.seed, workers=True)
    logger.info(f"Random seed set to {training_config.seed}")

    # Initialize data module
    logger.info("Initializing data module...")
    data_module = FrictionSegNetDataModule(
        dataset_factory=create_cityscapes_dataloaders,
        dataset_config=dataset_config,
        training_config=training_config,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = FrictionSegNetLightning(
        num_classes=NUM_CITYSCAPES_CLASSES,
        training_config=training_config,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config.checkpoint_dir,
        filename="frictionsegnet-{epoch:02d}-{val/mIoU:.4f}",
        monitor=training_config.monitor_metric,
        mode=training_config.monitor_mode,
        save_top_k=training_config.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(
        f"Checkpoint callback: monitoring {training_config.monitor_metric} ({training_config.monitor_mode})"
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=training_config.monitor_metric,
        mode=training_config.monitor_mode,
        patience=training_config.early_stop_patience,
        min_delta=training_config.early_stop_min_delta,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    logger.info(f"Early stopping: patience={training_config.early_stop_patience}")

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Rich progress bar for better visualization
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    # Setup logger
    if training_config.entity or training_config.project_name:
        wandb_logger = WandbLogger(
            project=training_config.project_name,
            entity=training_config.entity,
            name=training_config.run_name,
            log_model=False,  # We handle checkpointing ourselves
            config=asdict(training_config),
        )
        logger.info(f"WandB logger initialized: {training_config.project_name}")
        pl_logger = wandb_logger
    else:
        pl_logger = True  # Use default Lightning logger
        logger.warning("No WandB configuration provided, using default logger")

    # Initialize trainer
    logger.info("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=training_config.log_every_n_steps,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        check_val_every_n_epoch=training_config.check_val_every_n_epoch,
        deterministic=True,  # For reproducibility
        enable_model_summary=True,
    )

    logger.info("Trainer configured:")
    logger.info(f"  - Max epochs: {training_config.epochs}")
    logger.info(f"  - Accelerator: {training_config.accelerator}")
    logger.info(f"  - Precision: {training_config.precision}")
    logger.info(f"  - Gradient clip: {training_config.gradient_clip_val}")
    logger.info(
        f"  - Accumulate grad batches: {training_config.accumulate_grad_batches}"
    )

    # Start training
    logger.info("Starting training...")
    logger.info("=" * 80)

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=training_config.resume_from_checkpoint,
    )

    # Training completed
    logger.success("=" * 80)
    logger.success("Training completed!")
    logger.success(f"Best validation mIoU: {model.best_val_miou:.4f}")
    logger.success(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.success("=" * 80)

    return model, trainer
