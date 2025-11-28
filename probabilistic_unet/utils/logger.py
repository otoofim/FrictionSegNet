import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import threading


class LoggerSingleton:
    """
    Singleton class for managing a unified logger across the entire pipeline.

    This ensures:
    - Single logger instance throughout the application
    - Consistent configuration across all modules
    - Integration with WandB and TensorBoard
    - Thread-safe initialization
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Ensure only one instance exists (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger configuration (only once)."""
        # Skip if already initialized
        if LoggerSingleton._initialized:
            return

        with LoggerSingleton._lock:
            if LoggerSingleton._initialized:
                return

            # Remove default handler
            logger.remove()

            # Add console handler with beautiful formatting
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>",
                colorize=True,
                level="INFO",
                backtrace=True,
                diagnose=True,
            )

            # Add file handler with rotation
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            logger.add(
                log_dir / "training_{time:YYYY-MM-DD}.log",
                rotation="500 MB",
                retention="10 days",
                compression="zip",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                backtrace=True,
                diagnose=True,
            )

            # Add error-specific log file
            logger.add(
                log_dir / "errors_{time:YYYY-MM-DD}.log",
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
                backtrace=True,
                diagnose=True,
            )

            # Store references for external loggers
            self.wandb_run = None
            self.tensorboard_writer = None
            self.metrics_buffer = {}

            logger.info("Logger initialized successfully")
            LoggerSingleton._initialized = True

    def configure_wandb(self, wandb_run):
        """
        Configure WandB integration for logging metrics and artifacts.

        Args:
            wandb_run: Active WandB run object
        """
        self.wandb_run = wandb_run
        if wandb_run:
            logger.info(
                f"WandB logger configured: {wandb_run.project}/{wandb_run.name}"
            )
        else:
            logger.warning("WandB run is None, WandB logging disabled")

    def configure_tensorboard(self, log_dir: str = "logs/tensorboard"):
        """
        Configure TensorBoard integration for visualization.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = Path(log_dir)
            tb_log_dir.mkdir(parents=True, exist_ok=True)

            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard writer configured: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, install tensorboard package")
            self.tensorboard_writer = None

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to all configured backends (console, WandB, TensorBoard).

        Args:
            metrics: Dictionary of metric names and values
            step: Global step/iteration number
            commit: Whether to commit to WandB immediately
        """
        # Log to console
        metrics_str = ", ".join(
            [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]
        )
        logger.info(f"Metrics @ step {step}: {metrics_str}")

        # Log to WandB
        if self.wandb_run:
            try:
                if step is not None:
                    self.wandb_run.log(metrics, step=step, commit=commit)
                else:
                    self.wandb_run.log(metrics, commit=commit)
            except Exception as e:
                logger.error(f"Failed to log to WandB: {e}")

        # Log to TensorBoard
        if self.tensorboard_writer and step is not None:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.error(f"Failed to log to TensorBoard: {e}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters to all configured backends.

        Args:
            params: Dictionary of hyperparameter names and values
        """
        logger.info("Hyperparameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

        # Log to WandB
        if self.wandb_run:
            try:
                self.wandb_run.config.update(params, allow_val_change=True)
            except Exception as e:
                logger.error(f"Failed to log hyperparameters to WandB: {e}")

        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                # TensorBoard hparams requires special handling
                hparam_dict = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                }
                if hparam_dict:
                    self.tensorboard_writer.add_hparams(hparam_dict, {})
            except Exception as e:
                logger.error(f"Failed to log hyperparameters to TensorBoard: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """
        Log artifacts (models, datasets, etc.) to WandB.

        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        if self.wandb_run:
            try:
                import wandb

                artifact = wandb.Artifact(
                    name=Path(artifact_path).stem,
                    type=artifact_type,
                )
                artifact.add_file(artifact_path)
                self.wandb_run.log_artifact(artifact)
                logger.success(f"Artifact logged to WandB: {artifact_path}")
            except Exception as e:
                logger.error(f"Failed to log artifact to WandB: {e}")
        else:
            logger.warning("WandB not configured, artifact not logged")

    def log_image(
        self,
        tag: str,
        image,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ):
        """
        Log images to TensorBoard and WandB.

        Args:
            tag: Tag/name for the image
            image: Image tensor or numpy array
            step: Global step number
            caption: Optional caption for WandB
        """
        # Log to TensorBoard
        if self.tensorboard_writer and step is not None:
            try:
                import torch

                if isinstance(image, torch.Tensor):
                    self.tensorboard_writer.add_image(tag, image, step)
                else:
                    # Assume numpy array
                    import numpy as np

                    if isinstance(image, np.ndarray):
                        # Convert HWC to CHW if needed
                        if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                            image = np.transpose(image, (2, 0, 1))
                        self.tensorboard_writer.add_image(tag, image, step)
            except Exception as e:
                logger.error(f"Failed to log image to TensorBoard: {e}")

        # Log to WandB
        if self.wandb_run:
            try:
                import wandb

                wandb_image = wandb.Image(image, caption=caption)
                self.wandb_run.log({tag: wandb_image}, step=step)
            except Exception as e:
                logger.error(f"Failed to log image to WandB: {e}")

    def log_table(self, table_name: str, data: Dict[str, list]):
        """
        Log tabular data to WandB.

        Args:
            table_name: Name of the table
            data: Dictionary mapping column names to lists of values
        """
        if self.wandb_run:
            try:
                import wandb

                table = wandb.Table(
                    columns=list(data.keys()), data=list(zip(*data.values()))
                )
                self.wandb_run.log({table_name: table})
                logger.info(f"Table logged to WandB: {table_name}")
            except Exception as e:
                logger.error(f"Failed to log table to WandB: {e}")
        else:
            logger.warning("WandB not configured, table not logged")

    def close(self):
        """Close all logging handlers and writers."""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.error(f"Failed to close TensorBoard writer: {e}")

        if self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.error(f"Failed to finish WandB run: {e}")

    def get_logger(self):
        """Get the underlying loguru logger instance."""
        return logger


# Global singleton instance
_logger_singleton = None


def get_logger():
    """
    Get the singleton logger instance.

    Returns:
        LoggerSingleton: The singleton logger instance

    Usage:
        logger = get_logger()
        logger.get_logger().info("Message")

        # Or directly access loguru logger methods
        log = get_logger().get_logger()
        log.info("Hello")
    """
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = LoggerSingleton()
    return _logger_singleton


def get_loguru_logger():
    """
    Get the loguru logger instance directly (for convenience).

    Returns:
        Logger: The loguru logger instance

    Usage:
        from probabilistic_unet.utils.logger import get_loguru_logger

        logger = get_loguru_logger()
        logger.info("Training started")
    """
    return get_logger().get_logger()


# Convenience function for backward compatibility
def configure_logger(
    wandb_run=None,
    tensorboard_dir: Optional[str] = None,
):
    """
    Configure the logger with WandB and TensorBoard.

    Args:
        wandb_run: Active WandB run object
        tensorboard_dir: Directory for TensorBoard logs

    Usage:
        import wandb
        run = wandb.init(project="my-project")
        configure_logger(wandb_run=run, tensorboard_dir="logs/tensorboard")
    """
    logger_instance = get_logger()

    if wandb_run:
        logger_instance.configure_wandb(wandb_run)

    if tensorboard_dir:
        logger_instance.configure_tensorboard(tensorboard_dir)

    return logger_instance
