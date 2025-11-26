import sys

from probabilistic_unet.dataloader.cityscapes_loader import CityscapesDatasetConfig
from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from probabilistic_unet.dataloader.cityscapes_loader import (
    create_cityscapes_dataloaders,
)
from loguru import logger
from typing import Optional


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


class FrictionSegNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Cityscapes dataset.

    Encapsulates all data loading logic including train/val splits,
    augmentations, and DataLoader configuration.
    """

    def __init__(
        self,
        dataset_config: CityscapesDatasetConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.training_config = training_config

        # Override batch size and num_workers from training config
        self.dataset_config.batch_size = training_config.batch_size
        self.dataset_config.num_workers = training_config.num_workers

        logger.info(
            f"DataModule initialized with batch_size={training_config.batch_size}"
        )
        logger.info(f"Dataset root: {dataset_config.root_dir}")
        logger.info(f"Image size: {dataset_config.img_size}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        logger.info("Setting up datasets...")

        # Create dataloaders using the efficient Cityscapes system
        self.train_loader, self.val_loader = create_cityscapes_dataloaders(
            self.dataset_config
        )

        logger.success(f"✅ Training samples: {len(self.train_loader.dataset)}")
        logger.success(f"✅ Validation samples: {len(self.val_loader.dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self.val_loader
