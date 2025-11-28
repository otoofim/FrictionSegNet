from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import yaml
import os
from dotenv import load_dotenv


@dataclass
class DatasetConfig:
    """
    Base configuration class for segmentation datasets.

    This provides a standard interface for dataset configuration that can
    be extended by specific dataset configs. All parameters can be set
    via .env file or YAML configuration.
    """

    # Dataset paths
    root_dir: str = "./datasets"
    img_size: Tuple[int, int] = field(default_factory=lambda: (512, 1024))

    # Dataset splits
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # Augmentation settings
    use_augmentation: bool = True
    augmentation_seed: int = 200

    # DataLoader settings
    batch_size: int = 4
    num_workers: int = 4
    shuffle_train: bool = True
    shuffle_val: bool = False
    drop_last: bool = True
    device: str = "cuda"  # For pin_memory setting

    # Dataset-specific settings (e.g., Cityscapes)
    mode: str = "fine"  # 'fine' or 'coarse' for Cityscapes
    target_type: str = "semantic"  # 'semantic' or 'instance' for Cityscapes

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Ensure img_size is a tuple
        if isinstance(self.img_size, (list, tuple)):
            self.img_size = tuple(self.img_size)

        # Validate root directory exists or can be created
        root_path = Path(self.root_dir)
        if not root_path.exists():
            print(f"Warning: Dataset root directory does not exist: {self.root_dir}")

    @classmethod
    def from_yaml(
        cls, yaml_path: str, env_path: Optional[str] = ".env"
    ) -> "DatasetConfig":
        """
        Load configuration from YAML file with optional .env file.

        Args:
            yaml_path: Path to the YAML configuration file
            env_path: Path to the .env file (default: ".env" in current directory)

        Returns:
            DatasetConfig instance with loaded configuration

        Example YAML structure:
            root_dir: ${DATASET_ROOT}
            img_size: [512, 1024]
            batch_size: 4
            use_augmentation: true
        """
        # Load environment variables if .env file exists
        if env_path and Path(env_path).exists():
            load_dotenv(env_path)

        # Load YAML configuration
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Replace environment variable references
        config_dict = DatasetConfig._resolve_env_variables(config_dict)

        # Filter only valid fields for DatasetConfig
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DatasetConfig":
        """
        Create DatasetConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            DatasetConfig instance
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "DatasetConfig":
        """
        Create DatasetConfig from environment variables.

        Expected environment variables:
            DATASET_ROOT_DIR
            DATASET_IMG_SIZE (comma-separated, e.g., "512,1024")
            DATASET_TRAIN_SPLIT
            DATASET_VAL_SPLIT
            DATASET_TEST_SPLIT
            DATASET_USE_AUGMENTATION
            DATASET_AUGMENTATION_SEED
            DATASET_BATCH_SIZE
            DATASET_NUM_WORKERS
            DATASET_SHUFFLE_TRAIN
            DATASET_SHUFFLE_VAL
            DATASET_DROP_LAST
            DATASET_DEVICE

        Args:
            env_path: Path to the .env file

        Returns:
            DatasetConfig instance
        """
        if Path(env_path).exists():
            load_dotenv(env_path)

        config_dict = {}

        # Map environment variables to config fields
        if os.getenv("DATASET_ROOT_DIR"):
            config_dict["root_dir"] = os.getenv("DATASET_ROOT_DIR")

        if os.getenv("DATASET_IMG_SIZE"):
            img_size_str = os.getenv("DATASET_IMG_SIZE")
            config_dict["img_size"] = tuple(map(int, img_size_str.split(",")))

        if os.getenv("DATASET_TRAIN_SPLIT"):
            config_dict["train_split"] = os.getenv("DATASET_TRAIN_SPLIT")

        if os.getenv("DATASET_VAL_SPLIT"):
            config_dict["val_split"] = os.getenv("DATASET_VAL_SPLIT")

        if os.getenv("DATASET_TEST_SPLIT"):
            config_dict["test_split"] = os.getenv("DATASET_TEST_SPLIT")

        if os.getenv("DATASET_BATCH_SIZE"):
            config_dict["batch_size"] = int(os.getenv("DATASET_BATCH_SIZE"))

        if os.getenv("DATASET_NUM_WORKERS"):
            config_dict["num_workers"] = int(os.getenv("DATASET_NUM_WORKERS"))

        if os.getenv("DATASET_USE_AUGMENTATION"):
            config_dict["use_augmentation"] = (
                os.getenv("DATASET_USE_AUGMENTATION").lower() == "true"
            )

        if os.getenv("DATASET_AUGMENTATION_SEED"):
            config_dict["augmentation_seed"] = int(
                os.getenv("DATASET_AUGMENTATION_SEED")
            )

        if os.getenv("DATASET_SHUFFLE_TRAIN"):
            config_dict["shuffle_train"] = (
                os.getenv("DATASET_SHUFFLE_TRAIN").lower() == "true"
            )

        if os.getenv("DATASET_SHUFFLE_VAL"):
            config_dict["shuffle_val"] = (
                os.getenv("DATASET_SHUFFLE_VAL").lower() == "true"
            )

        if os.getenv("DATASET_DROP_LAST"):
            config_dict["drop_last"] = os.getenv("DATASET_DROP_LAST").lower() == "true"

        if os.getenv("DATASET_DEVICE"):
            config_dict["device"] = os.getenv("DATASET_DEVICE")

        # Dataset-specific settings
        if os.getenv("DATASET_MODE"):
            config_dict["mode"] = os.getenv("DATASET_MODE")

        if os.getenv("DATASET_TARGET_TYPE"):
            config_dict["target_type"] = os.getenv("DATASET_TARGET_TYPE")

        return cls(**config_dict)

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
    project_name: str = "Probabilistic-UNet"
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

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "TrainingConfig":
        """
        Create TrainingConfig from environment variables.

        Expected environment variables:
            MODEL_LATENT_DIM
            MODEL_BETA
            MODEL_NUM_SAMPLES
            MODEL_USE_POSTERIOR
            TRAINING_EPOCHS
            TRAINING_LEARNING_RATE
            TRAINING_WEIGHT_DECAY
            TRAINING_BATCH_SIZE
            TRAINING_ACCUMULATE_GRAD_BATCHES
            TRAINING_OPTIMIZER
            TRAINING_LR_SCHEDULER
            TRAINING_WARMUP_EPOCHS
            TRAINING_MIN_LR
            TRAINING_GRADIENT_CLIP_VAL
            TRAINING_LABEL_SMOOTHING
            TRAINING_VAL_EVERY_N_EPOCH
            TRAINING_CHECK_VAL_EVERY_N_EPOCH
            TRAINING_SAVE_TOP_K
            TRAINING_MONITOR_METRIC
            TRAINING_MONITOR_MODE
            TRAINING_EARLY_STOP_PATIENCE
            TRAINING_EARLY_STOP_MIN_DELTA
            WANDB_PROJECT
            WANDB_ENTITY
            WANDB_RUN_NAME
            TRAINING_LOG_EVERY_N_STEPS
            TRAINING_NUM_WORKERS
            TRAINING_SEED
            TRAINING_PRECISION
            TRAINING_ACCELERATOR
            TRAINING_DEVICES
            TRAINING_CHECKPOINT_DIR
            TRAINING_LOG_DIR
            TRAINING_RESUME_FROM_CHECKPOINT

        Args:
            env_path: Path to the .env file

        Returns:
            TrainingConfig instance
        """
        if Path(env_path).exists():
            load_dotenv(env_path)

        config_dict = {}

        # Model architecture
        if os.getenv("MODEL_LATENT_DIM"):
            config_dict["latent_dim"] = int(os.getenv("MODEL_LATENT_DIM"))
        if os.getenv("MODEL_BETA"):
            config_dict["beta"] = float(os.getenv("MODEL_BETA"))
        if os.getenv("MODEL_NUM_SAMPLES"):
            config_dict["num_samples"] = int(os.getenv("MODEL_NUM_SAMPLES"))
        if os.getenv("MODEL_USE_POSTERIOR"):
            config_dict["use_posterior"] = (
                os.getenv("MODEL_USE_POSTERIOR").lower() == "true"
            )

        # Training hyperparameters
        if os.getenv("TRAINING_EPOCHS"):
            config_dict["epochs"] = int(os.getenv("TRAINING_EPOCHS"))
        if os.getenv("TRAINING_LEARNING_RATE"):
            config_dict["learning_rate"] = float(os.getenv("TRAINING_LEARNING_RATE"))
        if os.getenv("TRAINING_WEIGHT_DECAY"):
            config_dict["weight_decay"] = float(os.getenv("TRAINING_WEIGHT_DECAY"))
        if os.getenv("TRAINING_BATCH_SIZE"):
            config_dict["batch_size"] = int(os.getenv("TRAINING_BATCH_SIZE"))
        if os.getenv("TRAINING_ACCUMULATE_GRAD_BATCHES"):
            config_dict["accumulate_grad_batches"] = int(
                os.getenv("TRAINING_ACCUMULATE_GRAD_BATCHES")
            )

        # Optimization
        if os.getenv("TRAINING_OPTIMIZER"):
            config_dict["optimizer"] = os.getenv("TRAINING_OPTIMIZER")
        if os.getenv("TRAINING_LR_SCHEDULER"):
            config_dict["lr_scheduler"] = os.getenv("TRAINING_LR_SCHEDULER")
        if os.getenv("TRAINING_WARMUP_EPOCHS"):
            config_dict["warmup_epochs"] = int(os.getenv("TRAINING_WARMUP_EPOCHS"))
        if os.getenv("TRAINING_MIN_LR"):
            config_dict["min_lr"] = float(os.getenv("TRAINING_MIN_LR"))

        # Regularization
        if os.getenv("TRAINING_GRADIENT_CLIP_VAL"):
            config_dict["gradient_clip_val"] = float(
                os.getenv("TRAINING_GRADIENT_CLIP_VAL")
            )
        if os.getenv("TRAINING_LABEL_SMOOTHING"):
            config_dict["label_smoothing"] = float(
                os.getenv("TRAINING_LABEL_SMOOTHING")
            )

        # Validation
        if os.getenv("TRAINING_VAL_EVERY_N_EPOCH"):
            config_dict["val_every_n_epoch"] = int(
                os.getenv("TRAINING_VAL_EVERY_N_EPOCH")
            )
        if os.getenv("TRAINING_CHECK_VAL_EVERY_N_EPOCH"):
            config_dict["check_val_every_n_epoch"] = int(
                os.getenv("TRAINING_CHECK_VAL_EVERY_N_EPOCH")
            )

        # Checkpointing
        if os.getenv("TRAINING_SAVE_TOP_K"):
            config_dict["save_top_k"] = int(os.getenv("TRAINING_SAVE_TOP_K"))
        if os.getenv("TRAINING_MONITOR_METRIC"):
            config_dict["monitor_metric"] = os.getenv("TRAINING_MONITOR_METRIC")
        if os.getenv("TRAINING_MONITOR_MODE"):
            config_dict["monitor_mode"] = os.getenv("TRAINING_MONITOR_MODE")

        # Early stopping
        if os.getenv("TRAINING_EARLY_STOP_PATIENCE"):
            config_dict["early_stop_patience"] = int(
                os.getenv("TRAINING_EARLY_STOP_PATIENCE")
            )
        if os.getenv("TRAINING_EARLY_STOP_MIN_DELTA"):
            config_dict["early_stop_min_delta"] = float(
                os.getenv("TRAINING_EARLY_STOP_MIN_DELTA")
            )

        # WandB logging
        if os.getenv("WANDB_PROJECT"):
            config_dict["project_name"] = os.getenv("WANDB_PROJECT")
        if os.getenv("WANDB_ENTITY"):
            wandb_entity = os.getenv("WANDB_ENTITY")
            config_dict["entity"] = wandb_entity if wandb_entity else None
        if os.getenv("WANDB_RUN_NAME"):
            wandb_run_name = os.getenv("WANDB_RUN_NAME")
            config_dict["run_name"] = wandb_run_name if wandb_run_name else None
        if os.getenv("TRAINING_LOG_EVERY_N_STEPS"):
            config_dict["log_every_n_steps"] = int(
                os.getenv("TRAINING_LOG_EVERY_N_STEPS")
            )

        # System
        if os.getenv("TRAINING_NUM_WORKERS"):
            config_dict["num_workers"] = int(os.getenv("TRAINING_NUM_WORKERS"))
        if os.getenv("TRAINING_SEED"):
            config_dict["seed"] = int(os.getenv("TRAINING_SEED"))
        if os.getenv("TRAINING_PRECISION"):
            config_dict["precision"] = os.getenv("TRAINING_PRECISION")
        if os.getenv("TRAINING_ACCELERATOR"):
            config_dict["accelerator"] = os.getenv("TRAINING_ACCELERATOR")
        if os.getenv("TRAINING_DEVICES"):
            config_dict["devices"] = int(os.getenv("TRAINING_DEVICES"))

        # Paths
        if os.getenv("TRAINING_CHECKPOINT_DIR"):
            config_dict["checkpoint_dir"] = os.getenv("TRAINING_CHECKPOINT_DIR")
        if os.getenv("TRAINING_LOG_DIR"):
            config_dict["log_dir"] = os.getenv("TRAINING_LOG_DIR")

        # Resume training
        if os.getenv("TRAINING_RESUME_FROM_CHECKPOINT"):
            resume_path = os.getenv("TRAINING_RESUME_FROM_CHECKPOINT")
            config_dict["resume_from_checkpoint"] = resume_path if resume_path else None

        return cls(**config_dict)

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
