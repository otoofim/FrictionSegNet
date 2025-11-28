from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from probabilistic_unet.utils.config_loader.config_dataclass import DatasetConfig
from probabilistic_unet.utils.logger import get_loguru_logger
from typing import Optional, Callable, Tuple


# Get singleton logger
logger = get_loguru_logger()


class FrictionSegNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation datasets.

    This DataModule serves as a bridge between the Lightning training loop and
    specific dataset implementations (Cityscapes, custom datasets, etc.).

    It supports:
    - Multiple dataset types through a factory pattern
    - Automatic configuration of batch size and num_workers
    - Flexible dataset creation via custom factory functions

    Usage:
        # For Cityscapes
        from probabilistic_unet.dataloader.cityscapes_loader import create_cityscapes_dataloaders

        datamodule = FrictionSegNetDataModule(
            dataset_factory=create_cityscapes_dataloaders,
            dataset_config=cityscapes_config,
            training_config=training_config
        )

        # For custom datasets
        datamodule = FrictionSegNetDataModule(
            dataset_factory=my_custom_dataloader_factory,
            dataset_config=my_config,
            training_config=training_config
        )
    """

    def __init__(
        self,
        dataset_factory: Callable[[DatasetConfig], Tuple[DataLoader, DataLoader]],
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
    ):
        """
        Initialize the DataModule with a dataset factory function.

        Args:
            dataset_factory: Function that takes a config and returns (train_loader, val_loader)
            dataset_config: Configuration for the dataset
            training_config: Training configuration including batch size and num_workers
        """
        super().__init__()
        self.dataset_factory = dataset_factory
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

        # Create dataloaders using the factory function
        self.train_loader, self.val_loader = self.dataset_factory(self.dataset_config)

        logger.success(f"✅ Training samples: {len(self.train_loader.dataset)}")
        logger.success(f"✅ Validation samples: {len(self.val_loader.dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self.val_loader
