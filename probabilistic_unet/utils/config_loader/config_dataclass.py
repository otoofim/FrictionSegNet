from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import yaml
import os
from dotenv import load_dotenv


@dataclass
class TrainingConfig:
    """Modern configuration for training with sensible defaults."""

    # Model architecture
    latent_dim: int = 6
    beta: float = 5.0
    num_samples: int = 16
    use_posterior: bool = True

    # Training hyperparameters
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 1

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    lr_scheduler: str = "cosine"  # cosine, plateau, step, none
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Regularization
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1

    # Validation
    val_every_n_epoch: int = 1
    check_val_every_n_epoch: int = 1

    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val/mIoU"
    monitor_mode: str = "max"

    # Early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.001

    # WandB logging
    project_name: str = "FrictionSegNet-Modern"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_every_n_steps: int = 10

    # System
    num_workers: int = 4
    seed: int = 42
    precision: str = "16-mixed"  # 32, 16-mixed, bf16-mixed
    accelerator: str = "auto"  # auto, gpu, cpu
    devices: int = 1

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(
        cls, yaml_path: str, env_path: Optional[str] = ".env"
    ) -> "TrainingConfig":
        """
        Load configuration from YAML file with optional .env file for sensitive data.

        Args:
            yaml_path: Path to the YAML configuration file
            env_path: Path to the .env file (default: ".env" in current directory)

        Returns:
            TrainingConfig instance with loaded configuration

        Example YAML structure:
            latent_dim: 6
            beta: 5.0
            epochs: 100
            learning_rate: 1e-4
            project_name: ${WANDB_PROJECT}  # Can reference env variables
            entity: ${WANDB_ENTITY}
        """
        # Load environment variables if .env file exists
        if env_path and Path(env_path).exists():
            load_dotenv(env_path)

        # Load YAML configuration
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Replace environment variable references in format ${VAR_NAME}
        config_dict = cls._resolve_env_variables(config_dict)

        # Filter only valid fields for TrainingConfig
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """
        Create TrainingConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            TrainingConfig instance
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)

    @staticmethod
    def _resolve_env_variables(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve environment variable references in config dictionary.
        Supports ${VAR_NAME} syntax.

        Args:
            config_dict: Dictionary potentially containing env variable references

        Returns:
            Dictionary with resolved environment variables
        """
        import re

        def resolve_value(value):
            if isinstance(value, str):
                # Match ${VAR_NAME} pattern
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, value)

                for match in matches:
                    env_value = os.getenv(match)
                    if env_value is not None:
                        value = value.replace(f"${{{match}}}", env_value)
                    else:
                        # Keep original if env variable not found
                        print(
                            f"Warning: Environment variable '{match}' not found, keeping original value"
                        )

                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config_dict)

    def to_yaml(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            output_path: Path where to save the YAML file
        """
        from dataclasses import asdict

        config_dict = asdict(self)

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        from dataclasses import asdict

        return asdict(self)
