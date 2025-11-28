"""
Config Loader Module

This module provides configuration classes for Probabilistic U-Net:
- TrainingConfig: Configuration for training parameters
- DatasetConfig: Base configuration for segmentation datasets
"""

from probabilistic_unet.utils.config_loader.config_dataclass import (
    TrainingConfig,
    DatasetConfig,
)

__all__ = ["TrainingConfig", "DatasetConfig"]
