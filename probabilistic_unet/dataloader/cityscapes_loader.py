"""
Cityscapes Dataset Loader with Efficient Augmentation System
============================================================

This module provides a comprehensive Cityscapes dataset loader with:
- Efficient augmentation system that multiplies dataset size
- 19 semantic classes for urban scene understanding
- Configurable image sizes and augmentation strategies
- Support for both training and validation splits

The augmentation system is particularly efficient as it treats each
augmentation as a separate sample, effectively multiplying the dataset
size by the number of augmentation strategies.
"""

import os
from torch.utils.data import DataLoader
import torch
from typing import Dict, Tuple, List, Optional

from probabilistic_unet.dataloader.base_segmentation_dataset import (
    BaseSegmentationDataset,
)
from probabilistic_unet.utils.config_loader.config_dataclass import DatasetConfig
from probabilistic_unet.utils.logger import get_loguru_logger

# Get singleton logger
logger = get_loguru_logger()


try:
    # Try to import official Cityscapes toolkit
    from cityscapesscripts.helpers.labels import labels

    # Use official Cityscapes labels
    CITYSCAPES_CLASSES = {
        label.trainId: label.name for label in labels if label.trainId != 255
    }
    NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_CLASSES)
    CITYSCAPES_COLORS = [label.color for label in labels if label.trainId != 255]

    logger.success("✅ Using official Cityscapes toolkit for labels and colors")

except ImportError:
    logger.warning(
        "⚠️  Official Cityscapes toolkit not found. Using fallback definitions."
    )
    logger.warning("   Install with: pip install cityscapesscripts")

    # Fallback definitions (19 classes for semantic segmentation)
    CITYSCAPES_CLASSES = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic_light",
        7: "traffic_sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }
    NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_CLASSES)
    CITYSCAPES_COLORS = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]


class CityscapesDataset(BaseSegmentationDataset):
    """
    Cityscapes Dataset with Efficient Augmentation System

    This dataset implementation inherits from BaseSegmentationDataset and provides
    Cityscapes-specific loading and configuration.

    Features:
    - Efficient augmentation system (dataset_size * num_augmentations)
    - Support for semantic and instance segmentation
    - Configurable image sizes
    - Proper handling of Cityscapes directory structure
    - Built-in visualization capabilities
    """

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int] = (512, 1024),
        split: str = "train",
        mode: str = "fine",
        target_type: str = "semantic",
        use_augmentation: bool = True,
        augmentation_seed: int = 200,
    ):
        """
        Initialize Cityscapes dataset.

        Args:
            root_dir: Path to Cityscapes dataset root
            img_size: Target image size (height, width)
            split: Dataset split ('train', 'val', 'test')
            mode: Annotation mode ('fine', 'coarse')
            target_type: Target type ('semantic', 'instance')
            use_augmentation: Whether to use augmentation
            augmentation_seed: Random seed for reproducible augmentation
        """
        # Store Cityscapes-specific attributes before parent init
        self.mode = mode
        self.target_type = target_type

        # Initialize parent class
        super().__init__(
            root_dir=root_dir,
            img_size=img_size,
            split=split,
            use_augmentation=use_augmentation,
            augmentation_seed=augmentation_seed,
        )

        # Set up directory paths
        self.images_dir = os.path.join(root_dir, "leftImg8bit", split)
        self.targets_dir = os.path.join(root_dir, f"gt{mode.capitalize()}", split)

        # Validate directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.targets_dir):
            raise FileNotFoundError(f"Targets directory not found: {self.targets_dir}")

        # Load image and target file paths
        self._load_file_paths()

        logger.info(f"Cityscapes {split} dataset loaded:")
        logger.info(f"  - Base samples: {len(self.images)}")
        logger.info(f"  - Augmentation strategies: {len(self.augmenters)}")
        logger.info(f"  - Total samples: {len(self)}")
        logger.info(f"  - Image size: {self.img_size}")

    def _load_file_paths(self):
        """Load image and target file paths from the dataset directory."""
        for city in sorted(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            if not os.path.isdir(img_dir) or not os.path.isdir(target_dir):
                continue

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_dir, file_name)

                    # Determine target file suffix
                    if self.target_type == "semantic":
                        target_suffix = "_gtFine_labelIds.png"
                    elif self.target_type == "instance":
                        target_suffix = "_gtFine_instanceIds.png"
                    else:
                        raise ValueError(f"Unsupported target_type: {self.target_type}")

                    target_name = file_name.replace("_leftImg8bit.png", target_suffix)
                    target_path = os.path.join(target_dir, target_name)

                    # Verify target file exists
                    if os.path.exists(target_path):
                        self.images.append(img_path)
                        self.targets.append(target_path)
                    else:
                        logger.warning(f"Target file not found: {target_path}")

    def get_num_classes(self) -> int:
        """Return number of classes in Cityscapes dataset."""
        return NUM_CITYSCAPES_CLASSES

    def get_class_names(self) -> Dict[int, str]:
        """Return class ID to name mapping."""
        return CITYSCAPES_CLASSES

    def get_class_colors(self) -> List[List[int]]:
        """Return class colors for visualization."""
        return CITYSCAPES_COLORS


def create_cityscapes_dataloaders(
    config: DatasetConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Cityscapes dataset.

    Args:
        config: CityscapesConfig object with dataset parameters

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CityscapesDataset(
        root_dir=config.root_dir,
        img_size=config.img_size,
        split=config.train_split,
        mode=config.mode,
        target_type=config.target_type,
        use_augmentation=config.use_augmentation,
        augmentation_seed=config.augmentation_seed,
    )

    val_dataset = CityscapesDataset(
        root_dir=config.root_dir,
        img_size=config.img_size,
        split=config.val_split,
        mode=config.mode,
        target_type=config.target_type,
        use_augmentation=False,  # No augmentation for validation
        augmentation_seed=config.augmentation_seed,
    )

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    return train_loader, val_loader


def visualize_cityscapes_sample(
    dataset: CityscapesDataset,
    sample: Dict[str, torch.Tensor],
    predicted_mask: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize a Cityscapes sample with image and segmentation mask.

    This is a convenience wrapper around the base class visualization method.

    Args:
        dataset: CityscapesDataset instance (needed for color mapping)
        sample: Dictionary containing 'image' and 'label' tensors
        predicted_mask: Optional predicted segmentation mask
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
    """
    dataset.visualize_sample(sample, predicted_mask, save_path, show_plot)


def print_dataset_statistics(dataset: CityscapesDataset) -> None:
    """
    Print comprehensive statistics about the Cityscapes dataset.

    This is a convenience wrapper around the base class statistics method.

    Args:
        dataset: CityscapesDataset instance
    """
    dataset.print_statistics()
