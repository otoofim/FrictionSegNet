from probabilistic_unet.utils.config_loader.config_dataclass import (
    TrainingConfig,
    DatasetConfig,
)
from probabilistic_unet.train_lightning import train_frictionsegnet
from probabilistic_unet.utils.logger import get_loguru_logger

# Get singleton logger
logger = get_loguru_logger()


def main():
    """Main entry point for training script."""

    # Load configurations from .env file
    logger.info("Loading configuration from .env file...")

    # Create dataset configuration from .env
    dataset_config = DatasetConfig.from_env()

    # Create training configuration from .env
    training_config = TrainingConfig.from_env()

    # Print configuration
    logger.info("Dataset Configuration:")
    logger.info(f"  Root: {dataset_config.root_dir}")
    logger.info(f"  Image size: {dataset_config.img_size}")
    logger.info(f"  Augmentation: {dataset_config.use_augmentation}")
    logger.info(f"  Batch size: {dataset_config.batch_size}")
    logger.info(f"  Num workers: {dataset_config.num_workers}")

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
    logger.info(f"  WandB Project: {training_config.project_name}")
    logger.info(f"  WandB Entity: {training_config.entity}")
    logger.info(f"  Run Name: {training_config.run_name}")

    # Start training
    model, trainer = train_frictionsegnet(dataset_config, training_config)

    logger.success("All done! 🎉")


if __name__ == "__main__":
    main()
