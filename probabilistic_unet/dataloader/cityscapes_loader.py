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
import zipfile
import shutil
import urllib.request
from pathlib import Path
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

# Cityscapes dataset download URLs (official dataset requires registration)
# These are placeholder URLs - users need to download manually from official site
CITYSCAPES_DOWNLOAD_INFO = {
    "official_url": "https://www.cityscapes-dataset.com/downloads/",
    "required_files": [
        "leftImg8bit_trainvaltest.zip",  # Images (11GB)
        "gtFine_trainvaltest.zip",  # Fine annotations (241MB)
    ],
    "note": "Cityscapes dataset requires registration and manual download from the official website.",
}


def _download_with_progress(url: str, destination: str) -> None:
    """
    Download a file with progress reporting.

    Args:
        url: URL to download from
        destination: Local path to save the file
    """

    def _progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        logger.info(f"Downloading: {percent}% complete")

    logger.info(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination, _progress_hook)
    logger.success(f"Download complete: {destination}")


def _extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file to a destination directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
    """
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.success(f"Extraction complete: {extract_to}")


def _verify_cityscapes_structure(root_dir: str, split: str, mode: str) -> bool:
    """
    Verify that the Cityscapes dataset has the expected directory structure.

    Args:
        root_dir: Root directory of the dataset
        split: Dataset split to verify ('train', 'val', 'test')
        mode: Annotation mode ('fine', 'coarse')

    Returns:
        True if structure is valid, False otherwise
    """
    images_dir = os.path.join(root_dir, "leftImg8bit", split)
    targets_dir = os.path.join(root_dir, f"gt{mode.capitalize()}", split)

    if not os.path.exists(images_dir):
        logger.warning(f"Missing images directory: {images_dir}")
        return False

    if not os.path.exists(targets_dir):
        logger.warning(f"Missing targets directory: {targets_dir}")
        return False

    # Check if directories have content
    if not os.listdir(images_dir):
        logger.warning(f"Images directory is empty: {images_dir}")
        return False

    if not os.listdir(targets_dir):
        logger.warning(f"Targets directory is empty: {targets_dir}")
        return False

    return True


def download_cityscapes_instructions(root_dir: str) -> None:
    """
    Print instructions for downloading the Cityscapes dataset.

    The Cityscapes dataset requires registration and cannot be automatically
    downloaded. This function provides clear instructions for users.

    Args:
        root_dir: Directory where the dataset should be placed
    """
    logger.error("❌ Cityscapes dataset not found!")
    logger.info("\n" + "=" * 70)
    logger.info("📥 CITYSCAPES DATASET DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 70)
    logger.info(f"\n{CITYSCAPES_DOWNLOAD_INFO['note']}")
    logger.info(f"\n1. Visit: {CITYSCAPES_DOWNLOAD_INFO['official_url']}")
    logger.info("2. Create an account and log in")
    logger.info("3. Download the following files:")
    for file in CITYSCAPES_DOWNLOAD_INFO["required_files"]:
        logger.info(f"   - {file}")
    logger.info(f"\n4. Extract the files to: {root_dir}")
    logger.info("\n5. Expected directory structure:")
    logger.info(f"   {root_dir}/")
    logger.info("   ├── leftImg8bit/")
    logger.info("   │   ├── train/")
    logger.info("   │   ├── val/")
    logger.info("   │   └── test/")
    logger.info("   ├── gtFine/")
    logger.info("   │   ├── train/")
    logger.info("   │   ├── val/")
    logger.info("   │   └── test/")
    logger.info("   └── gtCoarse/ (optional)")
    logger.info("\n" + "=" * 70 + "\n")


def setup_cityscapes_dataset(
    root_dir: str, split: str = "train", mode: str = "fine"
) -> bool:
    """
    Set up Cityscapes dataset by verifying or providing download instructions.

    Since Cityscapes requires manual download, this function checks if the
    dataset exists and provides instructions if it doesn't.

    Args:
        root_dir: Root directory for the dataset
        split: Dataset split to verify ('train', 'val', 'test')
        mode: Annotation mode ('fine', 'coarse')

    Returns:
        True if dataset is ready, False if manual download is needed
    """
    # Create root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # Check if dataset structure exists
    if _verify_cityscapes_structure(root_dir, split, mode):
        logger.success(f"✅ Cityscapes dataset found at: {root_dir}")
        return True

    # Dataset not found - provide instructions
    download_cityscapes_instructions(root_dir)
    return False


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
        auto_download: bool = True,
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
            auto_download: If True, check for dataset and provide download instructions if missing
        """
        # Store Cityscapes-specific attributes before parent init
        self.mode = mode
        self.target_type = target_type

        # Verify or setup dataset if auto_download is enabled
        if auto_download:
            if not setup_cityscapes_dataset(root_dir, split, mode):
                raise FileNotFoundError(
                    f"Cityscapes dataset not found at {root_dir}. "
                    "Please follow the instructions above to download the dataset manually."
                )

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

        # Final validation (should not fail if auto_download worked)
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
