"""
Utilities module for FrictionSegNet.

This module provides utility functions and classes including:
- Singleton logger with WandB/TensorBoard integration
- Configuration loaders
- Objective functions
- Support functions
"""

from probabilistic_unet.utils.logger import (
    get_logger,
    get_loguru_logger,
    configure_logger,
)

__all__ = [
    "get_logger",
    "get_loguru_logger",
    "configure_logger",
]
