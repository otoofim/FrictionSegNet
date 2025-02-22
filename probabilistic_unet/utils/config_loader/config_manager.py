import yaml
import json

import torch
from pathlib import Path
from typing import Tuple, Any, Dict
from dataclasses import asdict

from probabilistic_unet.utils.config_loader.config_class import (
    ContinueTrainingConfig,
    CustomizedGECOConfig,
    DatasetConfig,
    GECOConfig,
    PretrainedConfig,
    ProjectConfig,
)


class ConfigManager:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()

    def _load_and_validate_config(self) -> ProjectConfig:
        """Load and validate configuration from YAML file."""
        with open(self.config_path, "r") as file:
            self.config_dict = yaml.safe_load(file)

        # Create nested dataclass instances
        dataset_config = DatasetConfig(**self.config_dict["datasetConfig"])
        pretrained_config = PretrainedConfig(**self.config_dict["pretrained"])
        continue_tra_config = ContinueTrainingConfig(**self.config_dict["continue_tra"])
        customized_geco_config = CustomizedGECOConfig(
            **self.config_dict["customized_GECO"]
        )
        geco_config = GECOConfig(**self.config_dict["GECO"])

        # Remove nested configs from main dict and add them back as objects
        for key in [
            "datasetConfig",
            "pretrained",
            "continue_tra",
            "customized_GECO",
            "GECO",
        ]:
            self.config_dict.pop(key)

        return ProjectConfig(
            **self.config_dict,
            datasetConfig=dataset_config,
            pretrained=pretrained_config,
            continue_tra=continue_tra_config,
            customized_GECO=customized_geco_config,
            GECO=geco_config,
        )

    def get_device(self) -> torch.device:
        """Get PyTorch device based on configuration."""
        if self.config.device.lower() == "gpu" and torch.cuda.is_available():
            return torch.device(self.config.device_name)
        return torch.device("cpu")

    def get_input_dimensions(self) -> Tuple[int, int]:
        """Get input image dimensions."""
        return tuple(self.config.datasetConfig.input_img_dim)

    def is_geco_enabled(self) -> bool:
        """Check if either GECO mode is enabled."""
        return self.config.GECO.enable or self.config.customized_GECO.enable

    def get_dataset_path(self) -> Path:
        """Get the Volvo dataset path."""
        return Path(self.config.datasetConfig.volvoRootPath)

    def get_training_params(self) -> dict:
        """Get main training parameters."""
        return {
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "momentum": self.config.momentum,
            "beta": self.config.beta,
        }

    def should_load_pretrained(self) -> bool:
        """Check if should load pretrained model."""
        return self.config.pretrained.enable

    def should_continue_training(self) -> bool:
        """Check if should continue training from checkpoint."""
        return self.config.continue_tra.enable

    def _path_to_str(self, obj: Any) -> Any:
        """Convert Path objects to strings recursively in a dictionary."""
        if isinstance(obj, dict):
            return {key: self._path_to_str(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._path_to_str(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with Path objects as strings."""
        # Convert dataclass to dict
        config_dict = asdict(self.config)
        # Convert all Path objects to strings
        return self._path_to_str(config_dict)

    def get_json(self, indent: int = 2) -> str:
        """
        Get configuration as a JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2)

        Returns:
            str: JSON representation of the configuration
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=indent)
