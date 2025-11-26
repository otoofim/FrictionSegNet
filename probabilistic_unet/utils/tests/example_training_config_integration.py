"""
Example of integrating TrainingConfig with PyTorch Lightning training.

This shows how to use the YAML-based configuration system with your training pipeline.
"""

from pathlib import Path
from probabilistic_unet.utils.config_loader.config_dataclass import TrainingConfig


def example_training_integration():
    """
    Example showing how to integrate TrainingConfig with training code.
    """

    # Method 1: Load from YAML (recommended for experiments)
    config = TrainingConfig.from_yaml("config_training.yaml")

    # Method 2: Override specific parameters programmatically
    config = TrainingConfig.from_yaml("config_training.yaml")
    config.epochs = 50  # Override if needed
    config.batch_size = 8

    # Method 3: Create config for quick tests
    config = TrainingConfig(
        project_name="QuickTest",
        epochs=10,
        batch_size=2,
        num_workers=0,  # Useful for debugging
    )

    print("Configuration loaded:")
    print(f"  Project: {config.project_name}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Latent Dim: {config.latent_dim}")
    print(f"  Checkpoint Dir: {config.checkpoint_dir}")

    return config


def example_with_pytorch_lightning(config: TrainingConfig):
    """
    Example of using config with PyTorch Lightning.

    Note: This is a skeleton showing the pattern - actual implementation
    would require importing your model and dataloader classes.
    """

    # Example of how you might use it with Lightning Trainer
    # (requires pytorch_lightning to be imported)

    print("\nSetting up PyTorch Lightning training...")

    # Model initialization (example)
    # model = YourModel(
    #     latent_dim=config.latent_dim,
    #     beta=config.beta,
    #     learning_rate=config.learning_rate,
    #     weight_decay=config.weight_decay,
    # )

    # DataModule initialization (example)
    # datamodule = YourDataModule(
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    # )

    # Trainer setup (example)
    # trainer = pl.Trainer(
    #     max_epochs=config.epochs,
    #     accelerator=config.accelerator,
    #     devices=config.devices,
    #     precision=config.precision,
    #     gradient_clip_val=config.gradient_clip_val,
    #     accumulate_grad_batches=config.accumulate_grad_batches,
    #     val_check_interval=config.val_every_n_epoch,
    #     default_root_dir=config.checkpoint_dir,
    # )

    # Callbacks setup (example)
    # callbacks = [
    #     ModelCheckpoint(
    #         monitor=config.monitor_metric,
    #         mode=config.monitor_mode,
    #         save_top_k=config.save_top_k,
    #     ),
    #     EarlyStopping(
    #         monitor=config.monitor_metric,
    #         patience=config.early_stop_patience,
    #         min_delta=config.early_stop_min_delta,
    #     ),
    # ]

    # WandB logger setup (example)
    # logger = WandbLogger(
    #     project=config.project_name,
    #     entity=config.entity,
    #     name=config.run_name,
    # )

    print(f"  Would train for {config.epochs} epochs")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Monitoring: {config.monitor_metric} ({config.monitor_mode})")

    # Start training (example)
    # trainer.fit(model, datamodule=datamodule)


def example_save_experiment_config():
    """
    Example of saving experiment configuration for reproducibility.
    """

    # Create a config for an experiment
    config = TrainingConfig(
        project_name="Experiment_001",
        run_name="baseline_model",
        epochs=100,
        learning_rate=0.0001,
        batch_size=8,
        latent_dim=6,
    )

    # Save config for this specific experiment
    experiment_dir = Path("experiments") / config.run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = experiment_dir / "config.yaml"
    config.to_yaml(str(config_save_path))

    print(f"\nExperiment config saved to: {config_save_path}")
    print("This allows you to reproduce the experiment later!")


def example_config_sweep():
    """
    Example of creating multiple configurations for hyperparameter sweeps.
    """

    print("\nCreating configuration sweep...")

    # Define hyperparameter sweep
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [4, 8, 16]

    configs = []
    for lr in learning_rates:
        for bs in batch_sizes:
            config = TrainingConfig(
                project_name="Hyperparameter_Sweep",
                run_name=f"lr{lr}_bs{bs}",
                learning_rate=lr,
                batch_size=bs,
                epochs=50,
            )
            configs.append(config)

            # Save each config
            sweep_dir = Path("experiments/sweep")
            sweep_dir.mkdir(parents=True, exist_ok=True)
            config.to_yaml(str(sweep_dir / f"config_lr{lr}_bs{bs}.yaml"))

    print(f"Created {len(configs)} configurations for sweep")
    print("Configs saved to experiments/sweep/")


def main():
    """Run integration examples."""

    print("=" * 70)
    print("TrainingConfig Integration Examples")
    print("=" * 70)

    # Example 1: Basic integration
    print("\n1. Basic Configuration Loading")
    print("-" * 70)
    config = example_training_integration()

    # Example 2: Lightning integration pattern
    print("\n2. PyTorch Lightning Integration Pattern")
    print("-" * 70)
    example_with_pytorch_lightning(config)

    # Example 3: Save experiment config
    print("\n3. Saving Experiment Configuration")
    print("-" * 70)
    example_save_experiment_config()

    # Example 4: Config sweep
    print("\n4. Hyperparameter Sweep Configuration")
    print("-" * 70)
    example_config_sweep()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
