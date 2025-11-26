"""
Example usage of TrainingConfig with YAML and .env file loading.

This demonstrates three ways to create a TrainingConfig:
1. From a YAML file (with optional .env for sensitive data)
2. From a dictionary
3. Directly with default values
"""

from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig
from pathlib import Path


def example_1_load_from_yaml():
    """Load configuration from YAML file with environment variables."""
    print("=" * 60)
    print("Example 1: Loading from YAML with .env file")
    print("=" * 60)

    # Load config from YAML file
    # It will automatically load .env file if it exists
    config = TrainingConfig.from_yaml(
        yaml_path="config_training.yaml",
        env_path=".env",  # Optional, defaults to ".env"
    )

    print(f"Project Name: {config.project_name}")
    print(f"Entity: {config.entity}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print()

    return config


def example_2_load_from_dict():
    """Load configuration from a dictionary."""
    print("=" * 60)
    print("Example 2: Loading from Dictionary")
    print("=" * 60)

    config_dict = {
        "epochs": 50,
        "learning_rate": 0.0005,
        "batch_size": 8,
        "project_name": "MyProject",
        "entity": "my-team",
        "latent_dim": 8,
    }

    config = TrainingConfig.from_dict(config_dict)

    print(f"Project Name: {config.project_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print()

    return config


def example_3_default_config():
    """Create configuration with default values."""
    print("=" * 60)
    print("Example 3: Using Default Configuration")
    print("=" * 60)

    # Create with defaults, override specific values
    config = TrainingConfig(project_name="QuickExperiment", epochs=25, batch_size=16)

    print(f"Project Name: {config.project_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate} (default)")
    print()

    return config


def example_4_save_to_yaml():
    """Save configuration to YAML file."""
    print("=" * 60)
    print("Example 4: Saving Configuration to YAML")
    print("=" * 60)

    # Create a config
    config = TrainingConfig(
        project_name="SavedExperiment",
        epochs=75,
        learning_rate=0.0002,
    )

    # Save to YAML
    output_path = "saved_config.yaml"
    config.to_yaml(output_path)

    print(f"Configuration saved to: {output_path}")
    print()


def example_5_convert_to_dict():
    """Convert configuration to dictionary."""
    print("=" * 60)
    print("Example 5: Converting to Dictionary")
    print("=" * 60)

    config = TrainingConfig(project_name="DictExample")
    config_dict = config.to_dict()

    print("Configuration as dictionary:")
    for key, value in list(config_dict.items())[:5]:  # Show first 5 items
        print(f"  {key}: {value}")
    print(f"  ... ({len(config_dict)} total parameters)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "TrainingConfig Usage Examples" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Example 1: Load from YAML (will work if config_training.yaml exists)
    if Path("config_training.yaml").exists():
        try:
            config1 = example_1_load_from_yaml()
        except Exception as e:
            print(f"Could not load from YAML: {e}\n")
    else:
        print("Skipping Example 1: config_training.yaml not found\n")

    # Example 2: Load from dict
    config2 = example_2_load_from_dict()

    # Example 3: Default config
    config3 = example_3_default_config()

    # Example 4: Save to YAML
    example_4_save_to_yaml()

    # Example 5: Convert to dict
    example_5_convert_to_dict()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
