"""
Complete training script for FrictionSegNet - Probabilistic U-Net with VAE latent space sampling.
This script preserves all logging functionality and ensures comprehensive WandB integration.
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
from torch.utils.data import DataLoader, ConcatDataset
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
from probabilistic_unet.dataloader.generic_dataloader import classIds
from probabilistic_unet.dataloader.cityscapes_loader import CityscapesDataset

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


class TrainingLogger:
    """Handles all training and validation logging to WandB."""
    
    def __init__(self, class_names):
        self.class_names = class_names
        
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
    
    def log_images(self, batch, predictions, dataset, phase, epoch):
        """Log input images, ground truth, and predictions to WandB."""
        log_dict = {
            "epoch": epoch,
            f"input_{phase}": wandb.Image(batch["image"][-5:].detach().cpu()),
            f"ground_truth_{phase}": wandb.Image(batch["seg"][-5:].detach().cpu()),
        }
        
        if predictions.dim() == 5:  # Multiple samples from VAE
            # Average predictions across samples
            avg_predictions = torch.mean(predictions.detach().cpu(), 0)
            log_dict[f"prediction_{phase}"] = wandb.Image(
                dataset.prMask_to_color(avg_predictions[-5:])
            )
        else:
            log_dict[f"prediction_{phase}"] = wandb.Image(
                dataset.prMask_to_color(predictions[-5:].detach().cpu())
            )
        
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


def save_checkpoint(model, optimizer, epoch, tr_metrics, val_metrics, configs, checkpoint_type):
    """Save model checkpoint with all necessary information."""
    if not os.path.exists(configs.model_add):
        Path(configs.model_add).mkdir(parents=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tr_loss": tr_metrics,
        "val_loss": val_metrics,
        "hyper_params": json.dumps(configs.config_dict),
    }
    
    checkpoint_path = os.path.join(configs.model_add, f"{checkpoint_type}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train(configs: ConfigManager):
    """Main training function with comprehensive logging and VAE sampling preservation."""
    
    # Initialize WandB
    if configs.continue_tra.enable:
        wandb.init(
            config=json.dumps(configs.config_dict),
            project=configs.project_name,
            entity=configs.entity,
            name=configs.run_name,
            resume="must",
            id=configs.continue_tra.wandb_id,
        )
        print("WandB resumed...")
    else:
        wandb.init(
            config=asdict(configs),
            project=configs.project_name,
            entity=configs.entity,
            name=configs.run_name,
            resume="allow",
        )
    
    # Setup datasets
    tra_datasets = []
    val_datasets = []
    
    tra_datasets.append(
        CityscapesDataset(dataset_config=configs.datasetConfig, mode="train")
    )
    val_datasets.append(
        CityscapesDataset(dataset_config=configs.datasetConfig, mode="val")
    )
    print("Cityscapes dataset added!")
    
    train_dev_sets = ConcatDataset(tra_datasets)
    val_dev_sets = ConcatDataset(val_datasets)
    
    # Setup data loaders
    train_loader = DataLoader(
        dataset=train_dev_sets,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dev_sets,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # Setup device
    if configs.device == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    elif configs.device == "gpu":
        device = torch.device(configs.device_name)
        print("Running on the GPU")
    
    # Initialize model with VAE latent space sampling preserved
    model = ProUNet(
        gecoConfig=configs.GECO,
        num_classes=tra_datasets[0].get_num_classes(),
        LatentVarSize=configs.latent_dim,
        beta=configs.beta,
        training=True,
        num_samples=configs.num_samples,
        device=device,
    )
    
    # Load model state if continuing training or using pretrained
    if configs.continue_tra.enable:
        checkpoint = torch.load(
            configs.model_add / f"{configs.continue_tra.which_model}.pth",
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model state dict loaded...")
    elif configs.pretrained.enable:
        checkpoint = torch.load(
            configs.pretrained.model_add / f"{configs.pretrained.which_model}.pth",
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained model state dict loaded...")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs.learning_rate, weight_decay=configs.momentum
    )
    
    # Load optimizer state if continuing training
    if configs.continue_tra.enable:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state dict loaded...")
    
    # Initialize logger
    logger = TrainingLogger(classIds)
    
    # Setup training parameters
    best_val = -1
    val_every = 2
    wandb.watch(model)
    
    start_epoch = 0
    if configs.continue_tra.enable:
        start_epoch = checkpoint["epoch"] + 1
    
    end_epoch = configs.epochs
    total = configs.epochs
    
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
                train_batch, train_predictions, tra_datasets[0], "training", epoch + 1
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
                    val_batch, val_predictions, tra_datasets[0], "validation", epoch + 1
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
                    save_checkpoint(model, optimizer, epoch, tr_metrics, val_metrics, configs, "best")
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
    save_checkpoint(model, optimizer, end_epoch - 1, tr_metrics, val_metrics, configs, "final")


if __name__ == "__main__":
    # This would typically load configs from a configuration file
    # For now, we'll assume configs are passed to the train function
    print("Training script loaded. Call train(configs) to start training.")