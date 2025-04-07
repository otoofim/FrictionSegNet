import sys
from pathlib import Path
import warnings
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
import wandb

# Add local modules to path
sys.path.insert(1, "./architecture")
sys.path.insert(2, "./dataLoaders")

import torch.nn as nn
from probabilistic_unet.utils.config_loader.config_manager import ConfigManager
from probabilistic_unet.dataloader.GenericDataLoader import classIds
from probabilistic_unet.dataloader.CityscapesLoader import CityscapesLoader
from probabilistic_unet.model.ProUNet import ProUNet

# Suppress warnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def train(configs: ConfigManager) -> None:
    """
    Main training function for the ProUNet model.

    Args:
        configs (ConfigManager): Configuration object containing all training parameters
                               and model settings.

    The function handles:
    1. Wandb initialization and logging setup
    2. Dataset loading and preparation
    3. Model initialization and optimizer setup
    4. Training loop with validation
    5. Metrics calculation and logging
    6. Model checkpointing
    """
    # Setup model checkpoint directory
    configs.model_add = (
        Path("checkpoints") / f"{configs.project_name}_{configs.run_name}"
    )

    # Initialize wandb for experiment tracking
    _initialize_wandb(configs)

    # Prepare datasets and dataloaders
    train_loader, val_loader = _prepare_dataloaders(configs)

    # Setup device (CPU/GPU)
    device = _setup_device(configs)

    # Initialize model and optimizer
    model, optimizer = _initialize_model_and_optimizer(configs, device)

    # Initialize tracking dictionaries for losses and metrics
    tr_loss = _initialize_tracking_dict()
    val_loss = _initialize_tracking_dict()

    # Training loop
    start_epoch, total_epochs = _get_epoch_range(configs)

    with tqdm(
        range(start_epoch, total_epochs),
        initial=start_epoch,
        total=total_epochs,
        unit="epoch",
        leave=True,
        position=0,
    ) as epobar:
        for epoch in epobar:
            _run_training_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                tr_loss=tr_loss,
                configs=configs,
                epobar=epobar,
            )


def _initialize_wandb(configs: ConfigManager) -> None:
    """Initialize Weights & Biases logging."""
    if configs.continue_tra.enable:
        wandb.init(
            config=json.dumps(configs.config_dict),
            project=configs.project_name,
            entity=configs.entity,
            name=configs.run_name,
            resume="must",
            id=configs.continue_tra.wandb_id,
        )
    else:
        wandb.init(
            config=json.dumps(configs.config_dict),
            project=configs.project_name,
            entity=configs.entity,
            name=configs.run_name,
            resume="allow",
        )


def _prepare_dataloaders(configs: ConfigManager) -> tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation dataloaders.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Initialize dataset lists
    traDatasets = [CityscapesLoader(DatasetConfig=configs.datasetConfig, mode="train")]
    valDatasets = [CityscapesLoader(DatasetConfig=configs.datasetConfig, mode="val")]

    # Combine datasets
    train_dev_sets = ConcatDataset(traDatasets)
    val_dev_sets = ConcatDataset(valDatasets)

    # Create dataloaders
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

    return train_loader, val_loader


def _setup_device(configs: ConfigManager) -> torch.device:
    """Setup and return the appropriate device (CPU/GPU)."""
    if configs.device == "cpu":
        device = torch.device("cpu")
        print("Running on the CPU")
    else:
        device = torch.device(configs.device_name)
        print("Running on the GPU")
    return device


def _initialize_model_and_optimizer(
    configs: ConfigManager, device: torch.device
) -> tuple[ProUNet, torch.optim.Adam]:
    """
    Initialize the model and optimizer.

    Returns:
        tuple: (model, optimizer)
    """
    # Initialize model
    model = ProUNet(
        gecoConfig=configs.GECO,
        num_classes=configs.num_classes,
        LatentVarSize=configs.latent_dim,
        beta=configs.beta,
        training=True,
        num_samples=configs.num_samples,
        device=device,
    ).to(device)

    # Load model state if continuing training
    if configs.continue_tra.enable or configs.pretrained.enable:
        _load_model_state(model, configs, device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs.learning_rate, weight_decay=configs.momentum
    )

    # Load optimizer state if continuing training
    if configs.continue_tra.enable:
        _load_optimizer_state(optimizer, configs, device)

    return model, optimizer


def _run_training_epoch(
    epoch: int,
    model: ProUNet,
    optimizer: torch.optim.Adam,
    train_loader: DataLoader,
    device: torch.device,
    tr_loss: dict,
    configs: ConfigManager,
    epobar: tqdm,
) -> None:
    """
    Run a single training epoch.

    Args:
        epoch: Current epoch number
        model: The ProUNet model
        optimizer: The optimizer
        train_loader: Training data loader
        device: Device to run on
        tr_loss: Dictionary to track training losses
        configs: Configuration object
        epobar: Progress bar for epochs
    """
    model.train()
    epobar.set_description(f"Epoch {epoch + 1}")

    with tqdm(train_loader, unit="batch", leave=False) as batbar:
        for i, batch in enumerate(batbar):
            loss = _process_batch(model, optimizer, batch, device, tr_loss)

            # Update progress bar
            batbar.set_description(f"Batch {i + 1}")

            # Save checkpoint periodically
            if i % 1000 == 0:
                _save_checkpoint(model, optimizer, epoch, tr_loss, configs)

            # Log metrics to wandb
            _log_metrics(model, batch, tr_loss, epoch)


def _process_batch(
    model: ProUNet,
    optimizer: torch.optim.Adam,
    batch: dict,
    device: torch.device,
    tr_loss: dict,
) -> torch.Tensor:
    """
    Process a single batch of data.

    Returns:
        torch.Tensor: The loss value for this batch
    """
    optimizer.zero_grad()

    # Move batch to device
    batchImg = batch["image"].to(device)
    batchLabel = batch["label"].to(device)

    # Forward pass
    seg, priorDists, posteriorDists = model(batchImg, batchLabel)

    # Calculate losses
    loss, kl_mean, kl_losses, rec_loss, miou, ious, l1Loss, l2Loss, l3Loss, CM = (
        model.loss(batchLabel, seg, priorDists, posteriorDists)
    )

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    optimizer.step()

    # Update tracking metrics
    _update_tracking_metrics(
        tr_loss,
        loss,
        kl_mean,
        rec_loss,
        miou,
        l1Loss,
        l2Loss,
        l3Loss,
        CM,
        model,
        seg,
        batchLabel,
    )

    return loss


def _initialize_tracking_dict() -> dict:
    """Initialize and return the dictionary for tracking losses and metrics."""
    return {
        "Total training loss": [],
        "KL training loss": [],
        "Reconstruction training loss": [],
        "mIoU training": [],
        "ECE training": [],
        "MCE training": [],
        "RMSCE training": [],
        "lambda fri coeficient": [],
        "lambda seg coeficient": [],
        "NLL training": [],
        "Brie training": [],
        "Regression training loss": [],
        "Confusion Matrix training": [],
    }


"""
[Previous code remains the same up to _initialize_tracking_dict()]
"""


def _load_model_state(
    model: ProUNet, configs: ConfigManager, device: torch.device
) -> None:
    """
    Load model state from checkpoint.

    Args:
        model: The ProUNet model
        configs: Configuration object
        device: Device to load the state dict to
    """
    if configs.continue_tra.enable:
        checkpoint_path = configs.model_add / f"{configs.continue_tra.which_model}.pth"
    else:  # pretrained.enable
        checkpoint_path = (
            configs.pretrained.model_add / f"{configs.pretrained.which_model}.pth"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model state dict loaded successfully")


def _load_optimizer_state(
    optimizer: torch.optim.Adam, configs: ConfigManager, device: torch.device
) -> None:
    """
    Load optimizer state from checkpoint.

    Args:
        optimizer: The Adam optimizer
        configs: Configuration object
        device: Device to load the state dict to
    """
    checkpoint_path = configs.model_add / f"{configs.continue_tra.which_model}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Optimizer state dict loaded successfully")


def _get_epoch_range(configs: ConfigManager) -> tuple[int, int]:
    """
    Get the starting and total epochs for training.

    Returns:
        tuple: (start_epoch, total_epochs)
    """
    start_epoch = 0
    if configs.continue_tra.enable:
        checkpoint_path = configs.model_add / f"{configs.continue_tra.which_model}.pth"
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"] + 1

    return start_epoch, configs.epochs


def _save_checkpoint(
    model: ProUNet,
    optimizer: torch.optim.Adam,
    epoch: int,
    tr_loss: dict,
    configs: ConfigManager,
    val_loss: dict = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: The ProUNet model
        optimizer: The optimizer
        epoch: Current epoch number
        tr_loss: Training loss dictionary
        configs: Configuration object
        val_loss: Validation loss dictionary (optional)
    """
    # Create checkpoint directory if it doesn't exist
    if not configs.model_add.exists():
        configs.model_add.mkdir(parents=True)

    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tr_loss": tr_loss,
        "hyper_params": json.dumps(configs.config_dict),
    }

    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    # Save checkpoint
    torch.save(checkpoint, configs.model_add / "iterative.pth")


def _update_tracking_metrics(
    tr_loss: dict,
    loss: torch.Tensor,
    kl_mean: torch.Tensor,
    rec_loss: torch.Tensor,
    miou: torch.Tensor,
    l1Loss: torch.Tensor,
    l2Loss: torch.Tensor,
    l3Loss: torch.Tensor,
    CM: torch.Tensor,
    model: ProUNet,
    seg: torch.Tensor,
    batchLabel: torch.Tensor,
) -> None:
    """
    Update tracking metrics dictionary with current batch results.

    Args:
        tr_loss: Training loss dictionary
        loss: Total loss
        kl_mean: KL divergence mean
        rec_loss: Reconstruction loss
        miou: Mean IoU
        l1Loss: ECE loss
        l2Loss: MCE loss
        l3Loss: RMSCE loss
        CM: Confusion matrix
        model: The ProUNet model
        seg: Model predictions
        batchLabel: Ground truth labels
    """
    # Update basic metrics
    tr_loss["Total training loss"].append(loss.detach().cpu().item())
    tr_loss["KL training loss"].append(kl_mean.detach().cpu().item())
    tr_loss["Reconstruction training loss"].append(rec_loss.detach().cpu().item())
    tr_loss["mIoU training"].append(torch.mean(miou).detach().cpu().item())
    tr_loss["ECE training"].append(l1Loss.detach().cpu().item())
    tr_loss["MCE training"].append(l2Loss.detach().cpu().item())
    tr_loss["RMSCE training"].append(l3Loss.detach().cpu().item())

    # Update GECO coefficients if enabled
    if hasattr(model, "geco") and model.geco is not None:
        tr_loss["lambda seg coeficient"].append(
            model.geco.lambda_s.detach().cpu().item()
        )

    # Update NLL and Brier score
    tr_loss["NLL training"].append(
        nn.NLLLoss()(F.log_softmax(seg), torch.argmax(batchLabel, 1))
        .detach()
        .cpu()
        .item()
    )
    tr_loss["Brie training"].append(
        torch.mean(torch.square(seg - batchLabel)).detach().cpu().item()
    )
    tr_loss["Confusion Matrix training"].append(CM)


def _log_metrics(model: ProUNet, batch: dict, tr_loss: dict, epoch: int) -> None:
    """
    Log metrics to wandb.

    Args:
        model: The ProUNet model
        batch: Current batch of data
        tr_loss: Training loss dictionary
        epoch: Current epoch number
    """
    # Log images
    org_img = {
        "input": wandb.Image(batch["image"][-5:].detach().cpu()),
        "ground truth": wandb.Image(batch["seg"][-5:].detach().cpu()),
        "prediction": wandb.Image(
            model.dataloader.prMask_to_color(batch["seg"][-5:].detach().cpu())
        ),
    }
    wandb.log(org_img)

    # Log confusion matrix
    _log_confusion_matrix(tr_loss)

    # Log other metrics
    for key in tr_loss.keys():
        if "iou" in key or "mIoU" in key:
            wandb.log(
                {
                    key: torch.tensor([tr_loss[key]]).nanmean(),
                    "epoch": epoch + 1,
                }
            )
        elif "Confusion Matrix" not in key:
            wandb.log(
                {
                    key: torch.mean(torch.tensor(tr_loss[key])),
                    "epoch": epoch + 1,
                }
            )


def _log_confusion_matrix(tr_loss: dict) -> None:
    """
    Create and log confusion matrix visualization to wandb.

    Args:
        tr_loss: Training loss dictionary containing confusion matrix data
    """
    plt.rcParams.update({"font.size": 22})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig, ax = plt.subplots(figsize=(22, 20), dpi=100)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")

    # Calculate epoch-level confusion matrix
    CMEpoch = torch.nansum(torch.stack(tr_loss["Confusion Matrix training"]), dim=0)
    CMEpoch = torch.round(CMEpoch / CMEpoch.sum(dim=0, keepdim=True), decimals=2)
    CMEpoch = torch.nan_to_num(CMEpoch).cpu().detach().numpy()

    # Create and log confusion matrix plot
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=CMEpoch, display_labels=list(classIds.keys())
    ).plot(xticks_rotation=45, ax=ax)

    wandb.log({"Confusion matrix training": plt})
