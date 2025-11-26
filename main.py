import sys
from loguru import logger
from probabilistic_unet.dataloader.cityscapes_loader import CityscapesDatasetConfig
from probabilistic_unet.train import TrainingConfig
from probabilistic_unet.train_lightning import train_frictionsegnet


# Configure loguru for beautiful logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    level="INFO",
)
logger.add(
    "logs/training_{time}.log", rotation="500 MB", retention="10 days", level="DEBUG"
)


def main():
    """Main entry point for training script."""

    # Create dataset configuration
    dataset_config = CityscapesDatasetConfig()
    dataset_config.root_dir = "./datasets/Cityscapes"
    dataset_config.batch_size = 4  # Will be overridden by training_config
    dataset_config.img_size = (512, 1024)
    dataset_config.use_augmentation = True

    # Create training configuration
    training_config = TrainingConfig()
    training_config.epochs = 100
    training_config.learning_rate = 1e-4
    training_config.batch_size = 4
    training_config.latent_dim = 6
    training_config.beta = 5.0
    training_config.num_samples = 16
    training_config.precision = "16-mixed"  # Use mixed precision for speed
    training_config.gradient_clip_val = 1.0
    training_config.accumulate_grad_batches = 1

    # WandB configuration
    training_config.project_name = "FrictionSegNet-Modern"
    training_config.run_name = "cityscapes-lightning-v1"

    # Print configuration
    logger.info("Dataset Configuration:")
    logger.info(f"  Root: {dataset_config.root_dir}")
    logger.info(f"  Image size: {dataset_config.img_size}")
    logger.info(f"  Augmentation: {dataset_config.use_augmentation}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Epochs: {training_config.epochs}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Optimizer: {training_config.optimizer}")
    logger.info(f"  LR Scheduler: {training_config.lr_scheduler}")
    logger.info(f"  Precision: {training_config.precision}")
    logger.info(f"  Latent dim: {training_config.latent_dim}")
    logger.info(f"  Beta (KL weight): {training_config.beta}")
    logger.info(f"  Samples: {training_config.num_samples}")

    # Start training
    model, trainer = train_frictionsegnet(dataset_config, training_config)

    logger.success("All done! 🎉")


if __name__ == "__main__":
    main()
