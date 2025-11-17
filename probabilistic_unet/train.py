"""
Complete training script for FrictionSegNet - Probabilistic U-Net with VAE latent space sampling.
Specifically designed for Cityscapes dataset with comprehensive WandB logging.
"""

import sys
import os
import json
import warnings
from pathlib import Path
from statistics import mean
from collections import Counter
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import Normal, Independent, kl, MultivariateNormal

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchmetrics.classification import MulticlassCalibrationError

from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from sklearn import metrics

# Local imports
sys.path.insert(1, "./architecture")
sys.path.insert(2, "./dataLoaders")

from probabilistic_unet.model.pro_unet import ProUNet
from probabilistic_unet.utils.config_loader.config_manager import ConfigManager
from probabilistic_unet.dataloader.cityscapes_loader import (
    CityscapesDataset, CityscapesDatasetConfig, create_cityscapes_dataloaders,
    CITYSCAPES_CLASSES, NUM_CITYSCAPES_CLASSES, CITYSCAPES_COLORS
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


class TrainingLogger:
    """Handles all training and validation logging to WandB for Cityscapes dataset."""
    
    def __init__(self):
        self.class_names = CITYSCAPES_CLASSES
        
    def log_confusion_matrix(self, cm, title, epoch):
        """Log confusion matrix to WandB with proper formatting."""
        plt.rcParams.update({"font.size": 22})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        
        fig, ax = plt.subplots(figsize=(22, 20), dpi=100)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        
        # Normalize confusion matrix
        cm_normalized = torch.round(cm / cm.sum(dim=0, keepdim=True), decimals=2)
        cm_normalized = torch.nan_to_num(cm_normalized).cpu().detach().numpy()
        
        # Create confusion matrix display
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, 
            display_labels=list(self.class_names.values())
        ).plot(xticks_rotation=45, ax=ax)
        
        wandb.log({title: plt, "epoch": epoch})
        plt.close(fig)
    
    def log_images(self, batch, predictions, phase, epoch):
        """Log input images, ground truth, and predictions to WandB."""
        log_dict = {
            "epoch": epoch,
            f"input_{phase}": wandb.Image(batch["image"][-5:].detach().cpu()),
            f"ground_truth_{phase}": wandb.Image(batch["label"][-5:].detach().cpu()),
        }
        
        if predictions.dim() == 5:  # Multiple samples from VAE
            # Average predictions across samples
            avg_predictions = torch.mean(predictions.detach().cpu(), 0)
            log_dict[f"prediction_{phase}"] = wandb.Image(avg_predictions[-5:])
        else:
            log_dict[f"prediction_{phase}"] = wandb.Image(predictions[-5:].detach().cpu())
        
        wandb.log(log_dict)
    
    def log_metrics(self, metrics_dict, epoch):
        """Log all metrics to WandB."""
        log_dict = {"epoch": epoch}
        
        for key, values in metrics_dict.items():
            if "Confusion Matrix" in key:
                continue  # Skip confusion matrices, handled separately
            elif "iou" in key or "mIoU" in key:
                log_dict[key] = torch.tensor(values).nanmean()
            else:
                log_dict[key] = torch.mean(torch.tensor(values))
        
        wandb.log(log_dict)


def initialize_metrics_dict(phase):
    """Initialize metrics dictionary for training or validation."""
    return {
        f"Total {phase} loss": [],
        f"KL {phase} loss": [],
        f"Reconstruction {phase} loss": [],
        f"mIoU {phase}": [],
        f"ECE {phase}": [],
        f"MCE {phase}": [],
        f"RMSCE {phase}": [],
        f"NLL {phase}": [],
        f"Brie {phase}": [],
        f"Regression {phase} loss": [],
        f"Confusion Matrix {phase}": [],
    }


def add_layer_specific_metrics(metrics_dict, kl_losses, ious, phase):
    """Add layer-specific KL losses and IoU metrics."""
    for layer_id, kl_loss in kl_losses.items():
        key = f"KL {phase} loss layer{layer_id}"
        if key not in metrics_dict:
            metrics_dict[key] = []
        metrics_dict[key].append(torch.mean(kl_loss).detach().cpu().item())
    
    for class_id, iou in ious.items():
        key = f"iou of {class_id} {phase}"
        if key not in metrics_dict:
            metrics_dict[key] = []
        metrics_dict[key].append(torch.mean(iou).detach().cpu().item())


def train_epoch(model, train_loader, optimizer, device, logger, epoch, configs):
    """Execute one training epoch with comprehensive logging."""
    model.train()
    tr_metrics = initialize_metrics_dict("training")
    
    # Add GECO-specific metrics if enabled
    if configs.GECO.enable:
        tr_metrics["lambda seg coefficient"] = []
        tr_metrics["lambda fri coefficient"] = []
    
    with tqdm(train_loader, unit="batch", leave=False) as batbar:
        for i, batch in enumerate(batbar):
            batbar.set_description(f"Batch {i + 1}")
            optimizer.zero_grad()
            
            # Forward pass
            batch_img = batch["image"].to(device)
            batch_label = batch["label"].to(device)
            
            seg, prior_dists, posterior_dists = model(batch_img, batch_label)
            
            # Calculate losses
            (
                loss, kl_mean, kl_losses, rec_loss, miou, ious,
                l1_loss, l2_loss, l3_loss, cm
            ) = model.loss(batch_label, seg, prior_dists, posterior_dists)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            
            # Collect metrics
            tr_metrics["Total training loss"].append(loss.detach().cpu().item())
            tr_metrics["KL training loss"].append(kl_mean.detach().cpu().item())
            tr_metrics["Reconstruction training loss"].append(rec_loss.detach().cpu().item())
            tr_metrics["mIoU training"].append(torch.mean(miou).detach().cpu().item())
            tr_metrics["ECE training"].append(l1_loss.detach().cpu().item())
            tr_metrics["MCE training"].append(l2_loss.detach().cpu().item())
            tr_metrics["RMSCE training"].append(l3_loss.detach().cpu().item())
            tr_metrics["Confusion Matrix training"].append(cm)
            
            # Additional metrics
            tr_metrics["NLL training"].append(
                nn.NLLLoss()(F.log_softmax(seg, dim=1), torch.argmax(batch_label, 1))
                .detach().cpu().item()
            )
            tr_metrics["Brie training"].append(
                torch.mean(torch.square(seg - batch_label)).detach().cpu().item()
            )
            
            # GECO coefficients if enabled
            if configs.GECO.enable:
                tr_metrics["lambda seg coefficient"].append(
                    model.geco.lambda_s.detach().cpu().item()
                )
                if hasattr(model.geco, 'lambda_f'):
                    tr_metrics["lambda fri coefficient"].append(
                        model.geco.lambda_f.detach().cpu().item()
                    )
            
            # Add layer-specific metrics
            add_layer_specific_metrics(tr_metrics, kl_losses, ious, "training")
            
            # Periodic model saving
            if i % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, tr_metrics, {}, configs, "iterative")
    
    return tr_metrics, batch, seg


def validate_epoch(model, val_loader, device, logger, epoch, configs):
    """Execute one validation epoch with comprehensive logging."""
    model.eval()
    val_metrics = initialize_metrics_dict("validation")
    
    with torch.no_grad():
        with tqdm(val_loader, unit="batch", leave=False) as valbar:
            for i, batch in enumerate(valbar):
                valbar.set_description(f"Val_batch {i + 1}")
                
                batch_img = batch["image"].to(device)
                batch_label = batch["label"].to(device)
                fri_label = batch.get("FriLabel", torch.zeros_like(batch_label[:, :1])).to(device)
                
                # Get multiple samples for validation
                samples, priors, posteriors, fri_preds = model.evaluation(
                    batch_img, batch_label, fri_label
                )
                
                # Collect metrics across all samples
                loss_sum, kl_sum, rec_sum, iou_sum = [], [], [], []
                l1_sum, l2_sum, l3_sum, nll_sum, brie_sum, reg_sum = [], [], [], [], [], []
                kl_losses_sum, ious_sum = {}, {}
                
                for sample_idx, sample in enumerate(samples):
                    (
                        loss, kl_mean, kl_losses, rec_loss, miou, ious,
                        l1_loss, l2_loss, l3_loss, reg_loss, cm
                    ) = model.loss(
                        batch_label, sample, priors, posteriors,
                        fri_label, fri_preds[sample_idx] if fri_preds is not None else None
                    )
                    
                    # Accumulate metrics
                    loss_sum.append(loss)
                    kl_sum.append(kl_mean)
                    rec_sum.append(rec_loss)
                    iou_sum.append(miou)
                    l1_sum.append(l1_loss)
                    l2_sum.append(l2_loss)
                    l3_sum.append(l3_loss)
                    reg_sum.append(reg_loss)
                    
                    nll_sum.append(
                        nn.NLLLoss()(F.log_softmax(sample, dim=1), torch.argmax(batch_label, 1))
                    )
                    brie_sum.append(torch.mean(torch.square(sample - batch_label)))
                    
                    # Accumulate layer-specific metrics
                    for layer_id, kl_loss in kl_losses.items():
                        if layer_id not in kl_losses_sum:
                            kl_losses_sum[layer_id] = []
                        kl_losses_sum[layer_id].append(torch.mean(kl_loss).detach().cpu().item())
                    
                    for class_id, iou in ious.items():
                        if class_id not in ious_sum:
                            ious_sum[class_id] = []
                        ious_sum[class_id].append(torch.mean(iou).detach().cpu().item())
                
                # Average metrics across samples
                val_metrics["Total validation loss"].append(
                    torch.mean(torch.stack(loss_sum)).detach().cpu().item()
                )
                val_metrics["KL validation loss"].append(
                    torch.mean(torch.stack(kl_sum)).detach().cpu().item()
                )
                val_metrics["Reconstruction validation loss"].append(
                    torch.mean(torch.stack(rec_sum)).detach().cpu().item()
                )
                val_metrics["mIoU validation"].append(
                    torch.mean(torch.stack(iou_sum)).detach().cpu().item()
                )
                val_metrics["ECE validation"].append(
                    torch.mean(torch.stack(l1_sum)).detach().cpu().item()
                )
                val_metrics["MCE validation"].append(
                    torch.mean(torch.stack(l2_sum)).detach().cpu().item()
                )
                val_metrics["RMSCE validation"].append(
                    torch.mean(torch.stack(l3_sum)).detach().cpu().item()
                )
                val_metrics["NLL validation"].append(
                    torch.mean(torch.stack(nll_sum)).detach().cpu().item()
                )
                val_metrics["Brie validation"].append(
                    torch.mean(torch.stack(brie_sum)).detach().cpu().item()
                )
                val_metrics["Regression validation loss"].append(
                    torch.mean(torch.stack(reg_sum)).detach().cpu().item()
                )
                val_metrics["Confusion Matrix validation"].append(cm)
                
                # Add layer-specific validation metrics
                for layer_id, values in kl_losses_sum.items():
                    key = f"KL validation loss layer{layer_id}"
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(torch.mean(torch.tensor(values)).item())
                
                for class_id, values in ious_sum.items():
                    key = f"iou of {class_id} validation"
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(torch.tensor(values).nanmean().item())
    
    return val_metrics, batch, samples


def save_checkpoint(model, optimizer, epoch, tr_metrics, val_metrics, training_config, checkpoint_type):
    """Save model checkpoint with all necessary information."""
    model_dir = training_config.model_dir
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tr_loss": tr_metrics,
        "val_loss": val_metrics,
        "hyper_params": json.dumps(vars(training_config)),
    }
    
    checkpoint_path = os.path.join(model_dir, f"{checkpoint_type}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


class TrainingConfig:
    """Configuration class for training parameters only."""
    def __init__(self):
        # Training settings
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.val_every = 2
        
        # Model settings (VAE)
        self.latent_dim = 6
        self.beta = 1.0
        self.num_samples = 16
        
        # Device settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_name = 'cuda:0'
        
        # Logging settings
        self.project_name = 'FrictionSegNet-Cityscapes'
        self.entity = None
        self.run_name = 'cityscapes_training'
        
        # Checkpoint settings
        self.model_dir = './checkpoints'
        
        # GECO settings
        self.geco_enable = False


def train(dataset_config: CityscapesDatasetConfig, training_config: TrainingConfig):
    """Main training function for Cityscapes dataset with comprehensive logging and VAE sampling preservation."""
    
    # Initialize WandB
    config_dict = {**vars(dataset_config), **vars(training_config)}
    if hasattr(training_config, 'continue_tra') and training_config.continue_tra.enable:
        wandb.init(
            config=config_dict,
            project=training_config.project_name,
            entity=training_config.entity,
            name=training_config.run_name,
            resume="must",
            id=training_config.continue_tra.wandb_id,
        )
        print("WandB resumed...")
    else:
        wandb.init(
            config=config_dict,
            project=training_config.project_name,
            entity=training_config.entity,
            name=training_config.run_name,
            resume="allow",
        )
    
    # Create dataloaders using the enhanced system
    print("Creating Cityscapes dataloaders with efficient augmentation...")
    train_loader, val_loader = create_cityscapes_dataloaders(dataset_config)
    
    print(f"✅ Cityscapes dataloaders created!")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"🔄 Training augmentations: {len(train_loader.dataset.augmenters)}")
    print(f"🔄 Validation augmentations: {len(val_loader.dataset.augmenters)}")
    
    # Setup device
    if training_config.device == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    elif training_config.device == "gpu" or training_config.device == "cuda":
        device = torch.device(training_config.device_name if torch.cuda.is_available() else 'cpu')
        print(f"Running on device: {device}")
    else:
        device = torch.device("cpu")
        print("Default to CPU")
    
    # Initialize model with VAE latent space sampling preserved for Cityscapes
    geco_config = {'enable': training_config.geco_enable}
    model = ProUNet(
        gecoConfig=geco_config,
        num_classes=NUM_CITYSCAPES_CLASSES,
        LatentVarSize=training_config.latent_dim,
        beta=training_config.beta,
        training=True,
        num_samples=training_config.num_samples,
        device=device,
    )
    
    # Load model state if continuing training or using pretrained
    if hasattr(training_config, 'continue_tra') and training_config.continue_tra.enable:
        checkpoint = torch.load(
            os.path.join(training_config.model_dir, f"{training_config.continue_tra.which_model}.pth"),
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model state dict loaded...")
    elif hasattr(training_config, 'pretrained') and training_config.pretrained.enable:
        checkpoint = torch.load(
            os.path.join(training_config.pretrained.model_dir, f"{training_config.pretrained.which_model}.pth"),
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained model state dict loaded...")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config.learning_rate, 
        weight_decay=training_config.weight_decay
    )
    
    # Load optimizer state if continuing training
    if hasattr(training_config, 'continue_tra') and training_config.continue_tra.enable and 'checkpoint' in locals():
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state dict loaded...")
    
    # Initialize logger for Cityscapes
    logger = TrainingLogger()
    
    # Setup training parameters
    best_val = -1
    val_every = training_config.val_every
    wandb.watch(model)
    
    start_epoch = 0
    if hasattr(training_config, 'continue_tra') and training_config.continue_tra.enable and 'checkpoint' in locals():
        start_epoch = checkpoint["epoch"] + 1
    
    end_epoch = training_config.epochs
    total = end_epoch
    
    # Main training loop
    with tqdm(
        range(start_epoch, end_epoch),
        initial=start_epoch,
        total=total,
        unit="epoch",
        leave=True,
        position=0,
    ) as epobar:
        for epoch in epobar:
            epobar.set_description(f"Epoch {epoch + 1}")
            
            # Training phase
            tr_metrics, train_batch, train_predictions = train_epoch(
                model, train_loader, optimizer, device, logger, epoch, configs
            )
            
            # Log training metrics
            logger.log_metrics(tr_metrics, epoch + 1)
            
            # Log training images
            logger.log_images(
                train_batch, train_predictions, "training", epoch + 1
            )
            
            # Log training confusion matrix
            if tr_metrics["Confusion Matrix training"]:
                cm_epoch = torch.nansum(
                    torch.stack(tr_metrics["Confusion Matrix training"]), dim=0
                )
                logger.log_confusion_matrix(
                    cm_epoch, "Confusion matrix training", epoch + 1
                )
            
            # Validation phase
            if (epoch + 1) % val_every == 0:
                val_metrics, val_batch, val_predictions = validate_epoch(
                    model, val_loader, device, logger, epoch, configs
                )
                
                # Log validation metrics
                logger.log_metrics(val_metrics, epoch + 1)
                
                # Log validation images
                logger.log_images(
                    val_batch, val_predictions, "validation", epoch + 1
                )
                
                # Log validation confusion matrix
                if val_metrics["Confusion Matrix validation"]:
                    cm_epoch = torch.nansum(
                        torch.stack(val_metrics["Confusion Matrix validation"]), dim=0
                    )
                    logger.log_confusion_matrix(
                        cm_epoch, "Confusion matrix validation", epoch + 1
                    )
                
                # Save best model
                current_val_miou = torch.mean(torch.tensor(val_metrics["mIoU validation"]))
                if current_val_miou > best_val:
                    best_val = current_val_miou
                    save_checkpoint(model, optimizer, epoch, tr_metrics, val_metrics, training_config, "best")
                    print(f"New best validation mIoU: {best_val:.4f}")
            
            # Update progress bar
            epobar.set_postfix(
                ordered_dict={
                    "tr_loss": np.mean(tr_metrics["Total training loss"]),
                    "val_loss": np.mean(val_metrics.get("Total validation loss", [0])),
                }
            )
    
    print("Training completed!")
    print(f"Best validation mIoU: {best_val:.4f}")
    
    # Final model save
    save_checkpoint(model, optimizer, end_epoch - 1, tr_metrics, val_metrics, training_config, "final")


if __name__ == "__main__":
    # Example usage with separated configurations
    print("FrictionSegNet Training Script for Cityscapes Dataset")
    print("=" * 60)
    
    # Create dataset configuration (only dataset-related settings)
    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = './datasets/Cityscapes'
    dataset_config.batch_size = 4
    dataset_config.img_size = (512, 1024)
    dataset_config.use_augmentation = True
    
    # Create training configuration (only training-related settings)
    training_config = TrainingConfig()
    training_config.epochs = 100
    training_config.learning_rate = 1e-4
    training_config.latent_dim = 6
    training_config.num_samples = 16
    
    print("Dataset Configuration:")
    print(f"  Dataset root: {dataset_config.root_dir}")
    print(f"  Image size: {dataset_config.img_size}")
    print(f"  Batch size: {dataset_config.batch_size}")
    print(f"  Use augmentation: {dataset_config.use_augmentation}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Device: {training_config.device}")
    print(f"  VAE latent dim: {training_config.latent_dim}")
    print(f"  VAE samples: {training_config.num_samples}")
    print(f"  Model directory: {training_config.model_dir}")
    
    print(f"\nCityscapes classes: {NUM_CITYSCAPES_CLASSES}")
    
    print("\nTo start training, call:")
    print("  train(dataset_config, training_config)")
    
    print("\nExample with custom settings:")
    print("  dataset_config = CityscapesDatasetConfig()")
    print("  dataset_config.root_dir = '/path/to/cityscapes'")
    print("  dataset_config.batch_size = 8")
    print("  ")
    print("  training_config = TrainingConfig()")
    print("  training_config.epochs = 200")
    print("  training_config.learning_rate = 5e-4")
    print("  ")
    print("  train(dataset_config, training_config)")
    
    # Uncomment the line below to start training immediately
    # train(dataset_config, training_config)