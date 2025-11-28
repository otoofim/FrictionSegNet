"""
Dataloader module for FrictionSegNet

This module provides a flexible dataloader system for segmentation tasks with:
- BaseSegmentationDataset: Abstract parent class for all segmentation datasets
- CityscapesDataset: Cityscapes-specific implementation
- FrictionSegNetDataModule: PyTorch Lightning DataModule for training
"""

from probabilistic_unet.dataloader.base_segmentation_dataset import (
    BaseSegmentationDataset,
    prepare_aug_funcs,
)
from probabilistic_unet.utils.config_loader.config_dataclass import DatasetConfig
from probabilistic_unet.dataloader.cityscapes_loader import (
    CityscapesDataset,
    create_cityscapes_dataloaders,
    visualize_cityscapes_sample,
    print_dataset_statistics,
)
from probabilistic_unet.dataloader.lightning_dataloader import (
    FrictionSegNetDataModule,
)

__all__ = [
    "BaseSegmentationDataset",
    "DatasetConfig",
    "prepare_aug_funcs",
    "CityscapesDataset",
    "create_cityscapes_dataloaders",
    "visualize_cityscapes_sample",
    "print_dataset_statistics",
    "FrictionSegNetDataModule",
]
