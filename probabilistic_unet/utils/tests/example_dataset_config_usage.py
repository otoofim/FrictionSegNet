"""
Example: Loading DatasetConfig from different sources

This example demonstrates three ways to load dataset configuration:
1. From environment variables (.env file)
2. From YAML file with environment variable substitution
3. Programmatically with from_dict()
"""

from probabilistic_unet.utils.config_loader.config_dataclass import DatasetConfig
from probabilistic_unet.dataloader import CityscapesDatasetConfig
from pathlib import Path


def example_from_env():
    """Load configuration from environment variables."""
    print("=" * 60)
    print("Example 1: Loading from .env file")
    print("=" * 60)

    # This will load from .env file if it exists
    config = DatasetConfig.from_env(".env")

    print(f"Root dir: {config.root_dir}")
    print(f"Image size: {config.img_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers}")
    print(f"Use augmentation: {config.use_augmentation}")
    print(f"Device: {config.device}")
    print()

    return config


def example_from_yaml():
    """Load configuration from YAML file with environment variable substitution."""
    print("=" * 60)
    print("Example 2: Loading from YAML with env variable substitution")
    print("=" * 60)

    yaml_path = "config_dataset.yaml"

    if not Path(yaml_path).exists():
        print(f"Warning: {yaml_path} not found. Skipping this example.")
        return None

    # This will load from YAML and substitute ${VAR_NAME} with env variables
    config = DatasetConfig.from_yaml(yaml_path, env_path=".env")

    print(f"Root dir: {config.root_dir}")
    print(f"Image size: {config.img_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers}")
    print(f"Use augmentation: {config.use_augmentation}")
    print(f"Device: {config.device}")
    print()

    return config


def example_from_dict():
    """Load configuration from dictionary (programmatic)."""
    print("=" * 60)
    print("Example 3: Loading from dictionary (programmatic)")
    print("=" * 60)

    config_dict = {
        "root_dir": "./datasets/MyDataset",
        "img_size": [256, 512],
        "batch_size": 8,
        "num_workers": 2,
        "use_augmentation": True,
        "device": "cpu",
    }

    config = DatasetConfig.from_dict(config_dict)

    print(f"Root dir: {config.root_dir}")
    print(f"Image size: {config.img_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers}")
    print(f"Use augmentation: {config.use_augmentation}")
    print(f"Device: {config.device}")
    print()

    return config


def example_cityscapes_config():
    """Example using CityscapesDatasetConfig which extends DatasetConfig."""
    print("=" * 60)
    print("Example 4: Using CityscapesDatasetConfig")
    print("=" * 60)

    # CityscapesDatasetConfig extends DatasetConfig with Cityscapes-specific settings
    config = CityscapesDatasetConfig()

    print(f"Root dir: {config.root_dir}")
    print(f"Image size: {config.img_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Mode: {config.mode}")
    print(f"Target type: {config.target_type}")
    print()

    # You can also save it to YAML
    config.to_yaml("cityscapes_config_output.yaml")
    print("Saved configuration to cityscapes_config_output.yaml")
    print()

    return config


def example_integration_with_datamodule():
    """Example showing integration with ProbabilisticUNetDataModule."""
    print("=" * 60)
    print("Example 5: Integration with DataModule")
    print("=" * 60)

    from probabilistic_unet.dataloader import ProbabilisticUNetDataModule
    from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig

    # Load dataset config from YAML (with env variable substitution)
    if Path("config_dataset.yaml").exists() and Path(".env").exists():
        dataset_config = DatasetConfig.from_yaml("config_dataset.yaml", ".env")
        print("✅ Loaded dataset config from YAML with env variables")
    else:
        # Fallback to programmatic config
        dataset_config = DatasetConfig()
        dataset_config.root_dir = "./datasets/Cityscapes"
        print("⚠️  Using programmatic config (YAML or .env not found)")

    # Load training config
    training_config = TrainingConfig()
    training_config.batch_size = 4
    training_config.num_workers = 4

    print(f"\nDataset root: {dataset_config.root_dir}")
    print(f"Image size: {dataset_config.img_size}")
    print(f"Batch size: {training_config.batch_size}")
    print()

    # Note: Actual DataModule creation would happen during training
    # datamodule = ProbabilisticUNetDataModule.create_cityscapes(
    #     dataset_config, training_config
    # )

    print("DataModule configuration ready for training!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DatasetConfig Loading Examples")
    print("=" * 60 + "\n")

    # Run examples
    try:
        example_from_env()
    except Exception as e:
        print(f"Error in example_from_env: {e}\n")

    try:
        example_from_yaml()
    except Exception as e:
        print(f"Error in example_from_yaml: {e}\n")

    try:
        example_from_dict()
    except Exception as e:
        print(f"Error in example_from_dict: {e}\n")

    try:
        example_cityscapes_config()
    except Exception as e:
        print(f"Error in example_cityscapes_config: {e}\n")

    try:
        example_integration_with_datamodule()
    except Exception as e:
        print(f"Error in example_integration_with_datamodule: {e}\n")

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
