"""
Example demonstrating the usage of FrictionSegNet's singleton logger.

This script shows how to:
1. Use the basic logger for console and file logging
2. Configure WandB integration
3. Configure TensorBoard integration
4. Log metrics, hyperparameters, and artifacts
5. Log images and tables
"""

import torch
import wandb
from probabilistic_unet.utils.logger import (
    get_loguru_logger,
    get_logger,
    configure_logger,
)


def example_basic_logging():
    """Example 1: Basic logging to console and files."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Logging")
    print("=" * 80 + "\n")

    # Get the loguru logger for regular logging
    logger = get_loguru_logger()

    # Different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Logs are automatically saved to:
    # - logs/training_YYYY-MM-DD.log (all levels)
    # - logs/errors_YYYY-MM-DD.log (errors only)

    logger.info("Logs saved to: logs/training_*.log")


def example_metrics_logging():
    """Example 2: Logging metrics without WandB/TensorBoard."""
    print("\n" + "=" * 80)
    print("Example 2: Metrics Logging (Console + Files Only)")
    print("=" * 80 + "\n")

    logger = get_loguru_logger()
    logger_singleton = get_logger()

    # Simulate training loop
    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}/3")

        for step in range(5):
            # Simulate metrics
            loss = 1.0 / (step + 1 + epoch * 5)
            accuracy = 0.5 + (step + epoch * 5) * 0.05

            # Log metrics (will show in console and save to files)
            logger_singleton.log_metrics(
                {
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "train/epoch": epoch + 1,
                },
                step=epoch * 5 + step,
            )

        logger.success(f"Epoch {epoch + 1} completed")


def example_with_wandb():
    """Example 3: Full integration with WandB and TensorBoard."""
    print("\n" + "=" * 80)
    print("Example 3: WandB and TensorBoard Integration")
    print("=" * 80 + "\n")

    logger = get_loguru_logger()

    # Initialize WandB (set mode="disabled" to skip actual logging)
    logger.info("Initializing WandB...")
    run = wandb.init(
        project="frictionsegnet-logger-example",
        name="test-run",
        mode="disabled",  # Set to "online" to actually log to WandB
        config={"learning_rate": 0.001, "batch_size": 16, "epochs": 100},
    )

    # Configure logger with WandB and TensorBoard
    logger_singleton = configure_logger(
        wandb_run=run, tensorboard_dir="logs/tensorboard_example"
    )

    # Log hyperparameters
    logger.info("Logging hyperparameters...")
    logger_singleton.log_hyperparameters(
        {
            "model": "ProUNet",
            "latent_dim": 64,
            "num_classes": 19,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "batch_size": 16,
        }
    )

    # Simulate training with metrics logging
    logger.info("Starting training simulation...")
    for epoch in range(3):
        # Training metrics
        train_loss = 1.0 / (epoch + 1)
        train_iou = 0.5 + epoch * 0.1

        logger_singleton.log_metrics(
            {
                "train/loss": train_loss,
                "train/iou": train_iou,
                "train/learning_rate": 0.001 * (0.9**epoch),
            },
            step=epoch,
        )

        # Validation metrics
        val_loss = train_loss * 0.8
        val_iou = train_iou + 0.05

        logger_singleton.log_metrics(
            {
                "val/loss": val_loss,
                "val/iou": val_iou,
            },
            step=epoch,
        )

        logger.success(
            f"Epoch {epoch + 1} - Train IoU: {train_iou:.3f}, Val IoU: {val_iou:.3f}"
        )

    # Log a sample image
    logger.info("Logging sample image...")
    sample_image = torch.rand(3, 224, 224)
    logger_singleton.log_image(
        "predictions/sample", sample_image, step=2, caption="Sample prediction"
    )

    # Log a table of results
    logger.info("Logging results table...")
    logger_singleton.log_table(
        "class_metrics",
        {
            "class": ["road", "car", "person", "building"],
            "iou": [0.95, 0.87, 0.76, 0.89],
            "accuracy": [0.98, 0.92, 0.84, 0.94],
        },
    )

    # Log an artifact (mock checkpoint file)
    logger.info("Creating and logging artifact...")
    checkpoint_path = "logs/mock_checkpoint.pth"
    torch.save({"epoch": 3, "model": "mock"}, checkpoint_path)
    logger_singleton.log_artifact(checkpoint_path, artifact_type="model")

    logger.success("Training simulation completed!")

    # Clean up
    logger_singleton.close()
    logger.info("Logger closed successfully")


def example_error_handling():
    """Example 4: Error logging with traceback."""
    print("\n" + "=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80 + "\n")

    logger = get_loguru_logger()

    logger.info("Testing error logging...")

    try:
        # Simulate an error
        _ = 1 / 0
    except ZeroDivisionError:
        # Log exception with full traceback
        logger.exception("An error occurred during computation")
        logger.error("Check logs/errors_*.log for full traceback")

    logger.info("Error logged successfully")


def example_context_logging():
    """Example 5: Contextual logging."""
    print("\n" + "=" * 80)
    print("Example 5: Contextual Logging")
    print("=" * 80 + "\n")

    logger = get_loguru_logger()

    # Add context to logs
    logger.info("Processing data...")

    with logger.contextualize(stage="preprocessing"):
        logger.info("Loading images")
        logger.info("Applying augmentation")
        logger.success("Preprocessing completed")

    with logger.contextualize(stage="training"):
        logger.info("Building model")
        logger.info("Starting training loop")
        logger.success("Training completed")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FrictionSegNet Logger Examples")
    print("=" * 80)

    try:
        # Run examples
        example_basic_logging()
        example_metrics_logging()
        example_error_handling()
        example_context_logging()

        # Note: WandB example is commented out by default
        # Uncomment to test WandB integration (requires API key)
        # example_with_wandb()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("Check the logs/ directory for log files")
        print("=" * 80 + "\n")

    except Exception:
        logger = get_loguru_logger()
        logger.exception("Failed to run examples")
        raise


if __name__ == "__main__":
    main()
