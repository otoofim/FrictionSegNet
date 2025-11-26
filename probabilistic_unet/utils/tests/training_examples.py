#!/usr/bin/env python3
"""
Example: Using the Modern Training System Programmatically
==========================================================

This script shows how to use the new training system from Python code
rather than the CLI. Useful for notebooks or custom scripts.
"""

from probabilistic_unet.train_lightning import (
    train_frictionsegnet,
    TrainingConfig,
    FrictionSegNetLightning,
)
from probabilistic_unet.dataloader.cityscapes_loader import (
    CityscapesDatasetConfig,
    NUM_CITYSCAPES_CLASSES,
)
from loguru import logger


def example_basic_training():
    """Example 1: Basic training with default settings."""
    logger.info("Example 1: Basic Training")

    # Dataset configuration
    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"
    dataset_config.img_size = (512, 1024)
    dataset_config.use_augmentation = True

    # Training configuration
    training_config = TrainingConfig()
    training_config.epochs = 50
    training_config.batch_size = 4
    training_config.learning_rate = 1e-4
    training_config.run_name = "example-basic"

    # Train
    model, trainer = train_frictionsegnet(dataset_config, training_config)
    logger.success("Training completed!")

    return model, trainer


def example_custom_training():
    """Example 2: Custom training with specific hyperparameters."""
    logger.info("Example 2: Custom Training")

    # Dataset configuration
    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"
    dataset_config.img_size = (512, 1024)
    dataset_config.batch_size = 8  # Will be overridden

    # Training configuration with custom settings
    training_config = TrainingConfig()

    # Model architecture
    training_config.latent_dim = 8  # Larger latent space
    training_config.beta = 7.0  # Higher KL weight
    training_config.num_samples = 32  # More samples for uncertainty

    # Training settings
    training_config.epochs = 150
    training_config.batch_size = 8
    training_config.learning_rate = 5e-4
    training_config.weight_decay = 1e-4
    training_config.optimizer = "adamw"
    training_config.lr_scheduler = "cosine"
    training_config.warmup_epochs = 10

    # System settings
    training_config.precision = "16-mixed"  # Use mixed precision
    training_config.gradient_clip_val = 0.5
    training_config.accumulate_grad_batches = 2

    # Logging
    training_config.project_name = "FrictionSegNet-Experiments"
    training_config.run_name = "custom-high-beta"

    # Train
    model, trainer = train_frictionsegnet(dataset_config, training_config)
    logger.success(f"Best validation mIoU: {model.best_val_miou:.4f}")

    return model, trainer


def example_multi_gpu_training():
    """Example 3: Multi-GPU training."""
    logger.info("Example 3: Multi-GPU Training")

    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"

    training_config = TrainingConfig()
    training_config.epochs = 100
    training_config.batch_size = 16  # Larger batch for multi-GPU
    training_config.accelerator = "gpu"
    training_config.devices = 2  # Use 2 GPUs
    training_config.precision = "16-mixed"
    training_config.num_workers = 8
    training_config.run_name = "multi-gpu-experiment"

    model, trainer = train_frictionsegnet(dataset_config, training_config)

    return model, trainer


def example_resume_training():
    """Example 4: Resume training from checkpoint."""
    logger.info("Example 4: Resume Training")

    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"

    training_config = TrainingConfig()
    training_config.epochs = 150  # Train for more epochs
    training_config.resume_from_checkpoint = "checkpoints/last.ckpt"
    training_config.run_name = "resumed-training"

    model, trainer = train_frictionsegnet(dataset_config, training_config)

    return model, trainer


def example_hyperparameter_sweep():
    """Example 5: Hyperparameter sweep."""
    logger.info("Example 5: Hyperparameter Sweep")

    results = {}

    # Sweep over different beta values
    for beta in [1.0, 3.0, 5.0, 7.0, 10.0]:
        logger.info(f"Training with beta={beta}")

        dataset_config = CityscapesDatasetConfig()
        dataset_config.root_dir = "./datasets/Cityscapes"

        training_config = TrainingConfig()
        training_config.epochs = 50
        training_config.beta = beta
        training_config.run_name = f"beta-sweep-{beta}"

        model, trainer = train_frictionsegnet(dataset_config, training_config)

        # Store results
        results[beta] = {
            "best_miou": model.best_val_miou,
            "checkpoint": trainer.checkpoint_callback.best_model_path,
        }

        logger.info(f"Beta {beta}: mIoU = {model.best_val_miou:.4f}")

    # Print summary
    logger.success("Hyperparameter Sweep Complete!")
    logger.info("Results:")
    for beta, result in results.items():
        logger.info(f"  Beta {beta}: mIoU = {result['best_miou']:.4f}")

    return results


def example_inference():
    """Example 6: Load trained model for inference."""
    logger.info("Example 6: Inference with Trained Model")

    import torch
    from pathlib import Path

    # Load checkpoint
    checkpoint_path = "checkpoints/frictionsegnet-epoch=50-val_mIoU=0.7500.ckpt"

    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Train a model first!")
        return

    # Load model from checkpoint
    model = FrictionSegNetLightning.load_from_checkpoint(
        checkpoint_path,
        num_classes=NUM_CITYSCAPES_CLASSES,
        training_config=TrainingConfig(),  # Use default config
    )
    model.eval()

    logger.success("Model loaded!")

    # Example inference
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 512, 1024)  # Batch of 1 image

        # Generate samples
        samples, priors = model.model.inference(dummy_input)
        logger.info(f"Generated {len(samples)} samples")
        logger.info(f"Sample shape: {samples[0].shape}")

    return model


def example_custom_callbacks():
    """Example 7: Training with custom callbacks."""
    logger.info("Example 7: Custom Callbacks")

    from pytorch_lightning.callbacks import Callback

    # Custom callback to log every N batches
    class CustomLoggingCallback(Callback):
        def __init__(self, log_every_n_batches=100):
            self.log_every_n_batches = log_every_n_batches

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % self.log_every_n_batches == 0:
                logger.info(f"Batch {batch_idx}: loss = {outputs['loss']:.4f}")

    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"

    training_config = TrainingConfig()
    training_config.epochs = 50
    training_config.run_name = "custom-callbacks"

    # Would need to modify train_frictionsegnet to accept custom callbacks
    # For now, this is just an example of the concept
    logger.info("This would use custom callbacks if implemented!")

    return None


def main():
    """Run examples based on user choice."""
    import sys

    examples = {
        "1": ("Basic Training", example_basic_training),
        "2": ("Custom Training", example_custom_training),
        "3": ("Multi-GPU Training", example_multi_gpu_training),
        "4": ("Resume Training", example_resume_training),
        "5": ("Hyperparameter Sweep", example_hyperparameter_sweep),
        "6": ("Inference", example_inference),
        "7": ("Custom Callbacks", example_custom_callbacks),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        logger.info("Available Examples:")
        for key, (name, _) in examples.items():
            logger.info(f"  {key}. {name}")
        logger.info("\nUsage: python examples.py <number>")
        logger.info("Example: python examples.py 1")
        return

    if choice in examples:
        name, func = examples[choice]
        logger.info(f"Running Example: {name}")
        logger.info("=" * 80)

        try:
            func()
            logger.success(f"Example '{name}' completed successfully!")
        except Exception as e:
            logger.error(f"Example failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        logger.error(f"Invalid choice: {choice}")
        logger.info("Available examples: 1-7")


if __name__ == "__main__":
    main()
