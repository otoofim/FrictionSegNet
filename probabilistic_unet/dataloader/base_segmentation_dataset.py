"""
Base Segmentation Dataset
=========================

This module provides a parent class for segmentation dataset loaders that serves
as a bridge between the Lightning DataModule and specific dataset implementations.

Features:
- On-the-fly augmentation system that multiplies dataset size
- Built-in visualization for images, labels, and generated masks
- Standard interface for all segmentation datasets
- Efficient augmentation indexing
"""

import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def prepare_aug_funcs(img_size: Tuple[int, int]) -> Dict[str, Dict[str, T.Compose]]:
    """
    Prepares a dictionary of torchvision augmentation pipelines.

    This function creates various augmentation strategies that can be applied
    to both images and masks. Each augmentation strategy returns a dictionary
    with 'image' and 'mask' transforms.

    Args:
        img_size: Target image size (height, width)

    Returns:
        Dictionary mapping augmentation names to transform dictionaries
    """

    def base(img_tfms=[], mask_tfms=[]):
        """Helper to create matched image and mask transforms."""
        return {
            "image": T.Compose(
                [
                    T.ToTensor(),
                    *img_tfms,
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
            "mask": T.Compose(
                [
                    T.ToTensor(),
                    *mask_tfms,
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
        }

    return {
        "org": base(),
        "crop": base(
            img_tfms=[T.CenterCrop(size=500)], mask_tfms=[T.CenterCrop(size=500)]
        ),
        "grayScale": base(img_tfms=[T.Grayscale(num_output_channels=3)]),
        "colorJitter": base(
            img_tfms=[
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            ]
        ),
        "gaussianBlur": base(img_tfms=[T.GaussianBlur(kernel_size=51, sigma=5)]),
        "rotation": base(
            img_tfms=[T.RandomRotation(degrees=30)],
            mask_tfms=[T.RandomRotation(degrees=30)],
        ),
        "elastic": base(
            img_tfms=[T.ElasticTransform(alpha=500.0)],
            mask_tfms=[T.ElasticTransform(alpha=500.0)],
        ),
        "invert": base(img_tfms=[T.RandomInvert(p=1.0)]),
        "solarize": base(img_tfms=[T.RandomSolarize(threshold=0.05, p=1.0)]),
        "augMix": {
            "image": T.Compose(
                [
                    T.AugMix(severity=10, mixture_width=10),
                    T.ToTensor(),
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
            "mask": T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
        },
        "posterize": base(img_tfms=[T.RandomPosterize(bits=2, p=1.0)]),
        "erasing": {
            "image": T.Compose(
                [
                    T.ToTensor(),
                    T.RandomErasing(
                        p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
                    ),
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
            "mask": T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(img_size),
                    T.ToPILImage(),
                ]
            ),
        },
    }


class BaseSegmentationDataset(Dataset, ABC):
    """
    Abstract base class for segmentation datasets.

    This class provides a standard interface and common functionality for all
    segmentation dataset implementations. Child classes must implement specific
    methods for loading their dataset's files.

    The augmentation system treats each augmentation as a separate sample,
    effectively multiplying the dataset size by the number of augmentation
    strategies. This is achieved through efficient indexing:
    - base_idx = idx // num_augmentations
    - aug_idx = idx % num_augmentations

    Attributes:
        root_dir: Path to dataset root directory
        img_size: Target image size (height, width)
        split: Dataset split ('train', 'val', 'test')
        use_augmentation: Whether to apply augmentations
        augmentation_seed: Random seed for reproducibility
        images: List of image file paths
        targets: List of target/mask file paths
        augmenters: Dictionary of augmentation transforms
    """

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int] = (512, 1024),
        split: str = "train",
        use_augmentation: bool = True,
        augmentation_seed: int = 200,
    ):
        """
        Initialize base segmentation dataset.

        Args:
            root_dir: Path to dataset root directory
            img_size: Target image size (height, width)
            split: Dataset split ('train', 'val', 'test')
            use_augmentation: Whether to apply augmentations
            augmentation_seed: Random seed for reproducibility
        """
        super().__init__()

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.use_augmentation = use_augmentation
        self.augmentation_seed = augmentation_seed

        # Initialize file lists (to be populated by child classes)
        self.images: List[str] = []
        self.targets: List[str] = []

        # Set up augmentation system
        if self.use_augmentation:
            self.augmenters = prepare_aug_funcs(self.img_size)
        else:
            # Only use original images without augmentation
            self.augmenters = {"org": prepare_aug_funcs(self.img_size)["org"]}

        # Set random seeds for reproducibility
        self._set_seeds(augmentation_seed)

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducible augmentation."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def _load_file_paths(self):
        """
        Load image and target file paths from the dataset directory.

        This method must be implemented by child classes to populate
        self.images and self.targets lists.
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """
        Return the number of classes in the dataset.

        Returns:
            Number of segmentation classes
        """
        pass

    @abstractmethod
    def get_class_names(self) -> Dict[int, str]:
        """
        Return class ID to name mapping.

        Returns:
            Dictionary mapping class IDs to class names
        """
        pass

    @abstractmethod
    def get_class_colors(self) -> List[List[int]]:
        """
        Return class colors for visualization.

        Returns:
            List of RGB color triplets for each class
        """
        pass

    def __len__(self) -> int:
        """Return total number of samples (base_samples * num_augmentations)."""
        return len(self.images) * len(self.augmenters)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        The efficient augmentation system works by:
        1. Using integer division to get the base image index
        2. Using modulo to get the augmentation strategy
        3. Applying the selected augmentation to the base image

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'image': Input image tensor [C, H, W]
                - 'label': Segmentation mask tensor [1, H, W]
                - 'aug_type': Name of augmentation applied
                - 'base_idx': Base image index
        """
        # Efficient augmentation indexing
        img_idx = idx // len(self.augmenters)
        aug_key = list(self.augmenters.keys())[idx % len(self.augmenters)]
        aug = self.augmenters[aug_key]

        # Load image and mask
        img = self._load_image(img_idx)
        mask = self._load_mask(img_idx)

        # Apply augmentation
        img = aug["image"](img)
        mask = aug["mask"](mask)

        # Convert to tensors
        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()

        return {
            "image": img_tensor.float(),
            "label": mask_tensor.float(),
            "aug_type": aug_key,
            "base_idx": img_idx,
        }

    def _load_image(self, idx: int) -> Image.Image:
        """
        Load an image from file.

        Args:
            idx: Index of the image to load

        Returns:
            PIL Image in RGB format
        """
        return Image.open(self.images[idx]).convert("RGB")

    def _load_mask(self, idx: int) -> Image.Image:
        """
        Load a segmentation mask from file.

        Args:
            idx: Index of the mask to load

        Returns:
            PIL Image in grayscale format
        """
        return Image.open(self.targets[idx]).convert("L")

    def get_base_sample(self, base_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get original (non-augmented) sample by base index.

        Args:
            base_idx: Index of the base image (0 to len(images)-1)

        Returns:
            Dictionary containing image and label tensors without augmentation
        """
        if base_idx >= len(self.images):
            raise IndexError(
                f"Base index {base_idx} out of range [0, {len(self.images)})"
            )

        aug = self.augmenters["org"]

        img = self._load_image(base_idx)
        mask = self._load_mask(base_idx)

        img = aug["image"](img)
        mask = aug["mask"](mask)

        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()

        return {
            "image": img_tensor.float(),
            "label": mask_tensor.float(),
            "aug_type": "org",
            "base_idx": base_idx,
        }

    def visualize_sample(
        self,
        sample: Dict[str, torch.Tensor],
        predicted_mask: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        return_fig: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Visualize a sample with image, ground truth mask, and optional prediction.

        Args:
            sample: Dictionary containing 'image' and 'label' tensors
            predicted_mask: Optional predicted segmentation mask [1, H, W] or [H, W]
            save_path: Optional path to save the visualization
            show_plot: Whether to display the plot
            return_fig: Whether to return the figure object (useful for WandB/TensorBoard logging)

        Returns:
            matplotlib.Figure if return_fig is True, otherwise None
        """
        num_plots = 3 if predicted_mask is None else 4
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        if num_plots == 3:
            axes = list(axes)

        # Original image
        img = sample["image"].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Ground truth mask (grayscale)
        gt_mask = sample["label"].squeeze().cpu().numpy()
        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=self.get_num_classes() - 1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # Colored ground truth mask
        colored_gt = self.mask_to_color(gt_mask)
        axes[2].imshow(colored_gt)
        axes[2].set_title("Colored Ground Truth")
        axes[2].axis("off")

        # Predicted mask (if provided)
        if predicted_mask is not None:
            pred_mask = predicted_mask.squeeze().cpu().numpy()
            colored_pred = self.mask_to_color(pred_mask)
            axes[3].imshow(colored_pred)
            axes[3].set_title("Predicted Mask")
            axes[3].axis("off")

        # Add augmentation info if available
        if "aug_type" in sample:
            fig.suptitle(f"Augmentation: {sample['aug_type']}", fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if return_fig:
            return fig

        if show_plot:
            plt.show()
        else:
            plt.close()

        return None

    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert segmentation mask to colored visualization.

        Args:
            mask: 2D numpy array with class indices

        Returns:
            3D numpy array with RGB colors
        """
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        colors = self.get_class_colors()
        for class_id, color in enumerate(colors):
            colored_mask[mask == class_id] = color

        return colored_mask

    def get_augmentation_info(self) -> Dict[str, str]:
        """
        Get information about available augmentation strategies.

        Returns:
            Dictionary mapping augmentation names to descriptions
        """
        return {
            "org": "Original image without augmentation",
            "crop": "Center crop to 500x500 pixels",
            "grayScale": "Convert to grayscale (3 channels)",
            "colorJitter": "Random brightness, contrast, saturation, hue changes",
            "gaussianBlur": "Gaussian blur with kernel size 51, sigma 5",
            "rotation": "Random rotation up to 30 degrees",
            "elastic": "Elastic transform with alpha 500",
            "invert": "Random color inversion",
            "solarize": "Random solarization with threshold 0.05",
            "augMix": "AugMix augmentation with severity 10",
            "posterize": "Random posterization to 2 bits",
            "erasing": "Random erasing with scale 0.02-0.33",
        }

    def print_statistics(self) -> None:
        """Print comprehensive statistics about the dataset."""
        print(f"\n{'=' * 60}")
        print(f"DATASET STATISTICS - {self.__class__.__name__}")
        print(f"{'=' * 60}")
        print(f"Dataset split: {self.split}")
        print(f"Root directory: {self.root_dir}")
        print(f"Image size: {self.img_size}")
        print(f"Use augmentation: {self.use_augmentation}")
        print(f"Base samples: {len(self.images)}")
        print(f"Augmentation strategies: {len(self.augmenters)}")
        print(f"Total samples: {len(self)}")
        print(f"Number of classes: {self.get_num_classes()}")

        print("\nAugmentation strategies:")
        aug_info = self.get_augmentation_info()
        for aug_name in self.augmenters.keys():
            print(f"  - {aug_name}: {aug_info.get(aug_name, 'Custom augmentation')}")

        print("\nClass names:")
        for class_id, class_name in self.get_class_names().items():
            print(f"  {class_id:2d}: {class_name}")

        print(f"{'=' * 60}\n")
