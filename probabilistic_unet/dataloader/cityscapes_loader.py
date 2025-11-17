"""
Cityscapes Dataset Loader with Efficient Augmentation System
============================================================

This module provides a comprehensive Cityscapes dataset loader with:
- Efficient augmentation system that multiplies dataset size
- 19 semantic classes for urban scene understanding
- Configurable image sizes and augmentation strategies
- Support for both training and validation splits

The augmentation system is particularly efficient as it treats each
augmentation as a separate sample, effectively multiplying the dataset
size by the number of augmentation strategies.
"""

import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


try:
    # Try to import official Cityscapes toolkit
    from cityscapesscripts.helpers.labels import labels, trainId2label
    
    # Use official Cityscapes labels
    CITYSCAPES_CLASSES = {label.trainId: label.name for label in labels if label.trainId != 255}
    NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_CLASSES)
    CITYSCAPES_COLORS = [label.color for label in labels if label.trainId != 255]
    
    print("✅ Using official Cityscapes toolkit for labels and colors")
    
except ImportError:
    print("⚠️  Official Cityscapes toolkit not found. Using fallback definitions.")
    print("   Install with: pip install cityscapesscripts")
    
    # Fallback definitions (19 classes for semantic segmentation)
    CITYSCAPES_CLASSES = {
        0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
        5: "pole", 6: "traffic_light", 7: "traffic_sign", 8: "vegetation",
        9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
        14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
    }
    NUM_CITYSCAPES_CLASSES = len(CITYSCAPES_CLASSES)
    CITYSCAPES_COLORS = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ]


class CityscapesDatasetConfig:
    """Configuration class specifically for Cityscapes dataset loading."""
    
    def __init__(self):
        # Dataset paths
        self.root_dir = './datasets/Cityscapes'
        self.img_size = (512, 1024)
        
        # Dataset splits
        self.train_split = 'train'
        self.val_split = 'val'
        self.test_split = 'test'
        
        # Dataset mode and target type
        self.mode = 'fine'  # 'fine' or 'coarse'
        self.target_type = 'semantic'  # 'semantic' or 'instance'
        
        # Augmentation settings
        self.use_augmentation = True
        self.augmentation_seed = 200
        
        # DataLoader settings
        self.batch_size = 4
        self.num_workers = 4
        self.shuffle_train = True
        self.shuffle_val = False
        self.drop_last = True


def prepare_aug_funcs(img_size: Tuple[int, int]) -> Dict[str, Dict[str, T.Compose]]:
    """Prepares a dictionary of torchvision augmentation pipelines."""

    def base(img_tfms=[], mask_tfms=[]):
        return {
            "image": T.Compose([
                T.ToTensor(),
                *img_tfms,
                T.Resize(img_size),
                T.ToPILImage(),
            ]),
            "mask": T.Compose([
                T.ToTensor(),
                *mask_tfms,
                T.Resize(img_size),
                T.ToPILImage(),
            ])
        }

    return {
        "org": base(),
        "crop": base(
            img_tfms=[T.CenterCrop(size=500)],
            mask_tfms=[T.CenterCrop(size=500)]
        ),
        "grayScale": base(
            img_tfms=[T.Grayscale(num_output_channels=3)]
        ),
        "colorJitter": base(
            img_tfms=[T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
        ),
        "gaussianBlur": base(
            img_tfms=[T.GaussianBlur(kernel_size=51, sigma=5)]
        ),
        "rotation": base(
            img_tfms=[T.RandomRotation(degrees=30)],
            mask_tfms=[T.RandomRotation(degrees=30)]
        ),
        "elastic": base(
            img_tfms=[T.ElasticTransform(alpha=500.0)],
            mask_tfms=[T.ElasticTransform(alpha=500.0)]
        ),
        "invert": base(
            img_tfms=[T.RandomInvert(p=1.0)]
        ),
        "solarize": base(
            img_tfms=[T.RandomSolarize(threshold=0.05, p=1.0)]
        ),
        "augMix": {
            "image": T.Compose([
                T.AugMix(severity=10, mixture_width=10),
                T.ToTensor(),
                T.Resize(img_size),
                T.ToPILImage(),
            ]),
            "mask": T.Compose([
                T.ToTensor(),
                T.Resize(img_size),
                T.ToPILImage(),
            ])
        },
        "posterize": base(
            img_tfms=[T.RandomPosterize(bits=2, p=1.0)]
        ),
        "erasing": {
            "image": T.Compose([
                T.ToTensor(),
                T.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                T.Resize(img_size),
                T.ToPILImage(),
            ]),
            "mask": T.Compose([
                T.ToTensor(),
                T.Resize(img_size),
                T.ToPILImage(),
            ])
        },
    }


class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset with Efficient Augmentation System
    
    This dataset implementation uses an efficient augmentation strategy where
    each augmentation type is treated as a separate sample, effectively
    multiplying the dataset size by the number of augmentation strategies.
    
    Features:
    - Efficient augmentation system (dataset_size * num_augmentations)
    - Support for semantic and instance segmentation
    - Configurable image sizes
    - Proper handling of Cityscapes directory structure
    - Built-in visualization capabilities
    """
    
    def __init__(self, 
                 root_dir: str, 
                 img_size: Tuple[int, int] = (512, 1024), 
                 split: str = 'train', 
                 mode: str = 'fine', 
                 target_type: str = 'semantic',
                 use_augmentation: bool = True,
                 augmentation_seed: int = 200):
        """
        Initialize Cityscapes dataset.
        
        Args:
            root_dir: Path to Cityscapes dataset root
            img_size: Target image size (height, width)
            split: Dataset split ('train', 'val', 'test')
            mode: Annotation mode ('fine', 'coarse')
            target_type: Target type ('semantic', 'instance')
            use_augmentation: Whether to use augmentation
            augmentation_seed: Random seed for reproducible augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.target_type = target_type
        self.img_size = img_size
        self.use_augmentation = use_augmentation

        # Set up directory paths
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, f'gt{mode.capitalize()}', split)

        # Validate directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.targets_dir):
            raise FileNotFoundError(f"Targets directory not found: {self.targets_dir}")

        # Initialize file lists
        self.images = []
        self.targets = []
        
        # Set up augmentation system
        if self.use_augmentation:
            self.augmenters = prepare_aug_funcs(self.img_size)
        else:
            # Only use original images without augmentation
            self.augmenters = {"org": prepare_aug_funcs(self.img_size)["org"]}

        # Load image and target file paths
        self._load_file_paths()

        # Set random seeds for reproducibility
        self._set_seeds(augmentation_seed)
        
        print(f"Cityscapes {split} dataset loaded:")
        print(f"  - Base samples: {len(self.images)}")
        print(f"  - Augmentation strategies: {len(self.augmenters)}")
        print(f"  - Total samples: {len(self)}")
        print(f"  - Image size: {self.img_size}")
        
    def _load_file_paths(self):
        """Load image and target file paths from the dataset directory."""
        for city in sorted(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            
            if not os.path.isdir(img_dir) or not os.path.isdir(target_dir):
                continue

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_dir, file_name)
                    
                    # Determine target file suffix
                    if self.target_type == 'semantic':
                        target_suffix = '_gtFine_labelIds.png'
                    elif self.target_type == 'instance':
                        target_suffix = '_gtFine_instanceIds.png'
                    else:
                        raise ValueError(f"Unsupported target_type: {self.target_type}")
                    
                    target_name = file_name.replace('_leftImg8bit.png', target_suffix)
                    target_path = os.path.join(target_dir, target_name)
                    
                    # Verify target file exists
                    if os.path.exists(target_path):
                        self.images.append(img_path)
                        self.targets.append(target_path)
                    else:
                        print(f"Warning: Target file not found: {target_path}")

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducible augmentation."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
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
            Dictionary containing 'image' and 'label' tensors
        """
        # Efficient augmentation indexing
        img_idx = idx // len(self.augmenters)
        aug_key = list(self.augmenters.keys())[idx % len(self.augmenters)]
        aug = self.augmenters[aug_key]

        # Load image and mask
        img = Image.open(self.images[img_idx]).convert("RGB")
        mask = Image.open(self.targets[img_idx]).convert("L")

        # Apply augmentation
        img = aug["image"](img)
        mask = aug["mask"](mask)

        # Convert to tensors
        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()

        return {
            "image": img_tensor.float(),
            "label": mask_tensor.float(),
            "aug_type": aug_key,  # Include augmentation type for debugging
            "base_idx": img_idx   # Include base image index for debugging
        }
    
    def get_base_sample(self, base_idx: int) -> Dict[str, torch.Tensor]:
        """Get original (non-augmented) sample by base index."""
        if base_idx >= len(self.images):
            raise IndexError(f"Base index {base_idx} out of range [0, {len(self.images)})")
            
        aug = self.augmenters["org"]
        
        img = Image.open(self.images[base_idx]).convert("RGB")
        mask = Image.open(self.targets[base_idx]).convert("L")
        
        img = aug["image"](img)
        mask = aug["mask"](mask)
        
        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()
        
        return {
            "image": img_tensor.float(),
            "label": mask_tensor.float(),
            "aug_type": "org",
            "base_idx": base_idx
        }
    
    def get_num_classes(self) -> int:
        """Return number of classes in Cityscapes dataset."""
        return NUM_CITYSCAPES_CLASSES
    
    def get_class_names(self) -> Dict[int, str]:
        """Return class ID to name mapping."""
        return CITYSCAPES_CLASSES
    
    def get_class_colors(self) -> List[List[int]]:
        """Return class colors for visualization."""
        return CITYSCAPES_COLORS


def create_cityscapes_dataloaders(config: CityscapesDatasetConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Cityscapes dataset.
    
    Args:
        config: CityscapesConfig object with dataset parameters
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CityscapesDataset(
        root_dir=config.root_dir,
        img_size=config.img_size,
        split=config.train_split,
        mode=config.mode,
        target_type=config.target_type,
        use_augmentation=config.use_augmentation,
        augmentation_seed=config.augmentation_seed
    )
    
    val_dataset = CityscapesDataset(
        root_dir=config.root_dir,
        img_size=config.img_size,
        split=config.val_split,
        mode=config.mode,
        target_type=config.target_type,
        use_augmentation=False,  # No augmentation for validation
        augmentation_seed=config.augmentation_seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_val,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    return train_loader, val_loader


def visualize_cityscapes_sample(sample: Dict[str, torch.Tensor], 
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
    """
    Visualize a Cityscapes sample with image and segmentation mask.
    
    Args:
        sample: Dictionary containing 'image' and 'label' tensors
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = sample['image'].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Segmentation mask (grayscale)
    mask = sample['label'].squeeze().cpu().numpy()
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=NUM_CITYSCAPES_CLASSES-1)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Colored segmentation mask
    colored_mask = mask_to_color(mask)
    axes[2].imshow(colored_mask)
    axes[2].set_title('Colored Segmentation')
    axes[2].axis('off')
    
    # Add augmentation info if available
    if 'aug_type' in sample:
        fig.suptitle(f"Augmentation: {sample['aug_type']}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask to colored visualization.
    
    Args:
        mask: 2D numpy array with class indices
        
    Returns:
        3D numpy array with RGB colors
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(CITYSCAPES_COLORS):
        colored_mask[mask == class_id] = color
    
    return colored_mask


def get_augmentation_info() -> Dict[str, str]:
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
        "erasing": "Random erasing with scale 0.02-0.33"
    }


def print_dataset_statistics(dataset: CityscapesDataset) -> None:
    """
    Print comprehensive statistics about the dataset.
    
    Args:
        dataset: CityscapesDataset instance
    """
    print(f"\n{'='*60}")
    print(f"CITYSCAPES DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Dataset split: {dataset.split}")
    print(f"Mode: {dataset.mode}")
    print(f"Target type: {dataset.target_type}")
    print(f"Image size: {dataset.img_size}")
    print(f"Use augmentation: {dataset.use_augmentation}")
    print(f"Base samples: {len(dataset.images)}")
    print(f"Augmentation strategies: {len(dataset.augmenters)}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    print(f"\nAugmentation strategies:")
    aug_info = get_augmentation_info()
    for aug_name in dataset.augmenters.keys():
        print(f"  - {aug_name}: {aug_info.get(aug_name, 'Custom augmentation')}")
    
    print(f"\nClass names:")
    for class_id, class_name in dataset.get_class_names().items():
        print(f"  {class_id:2d}: {class_name}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """
    Demo script showing the enhanced Cityscapes dataset functionality.
    """
    print("Cityscapes Dataset Loader Demo")
    print("=" * 50)
    
    # Create configuration
    config = CityscapesDatasetConfig()
    config.root_dir = "../../datasets/Cityscapes"  # Adjust path as needed
    config.batch_size = 2
    config.num_workers = 2
    
    print(f"Configuration:")
    print(f"  Root directory: {config.root_dir}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Use augmentation: {config.use_augmentation}")
    
    try:
        # Create datasets using the enhanced system
        print("\nCreating datasets...")
        train_loader, val_loader = create_cityscapes_dataloaders(config)
        
        # Get the underlying datasets
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Print statistics
        print_dataset_statistics(train_dataset)
        print_dataset_statistics(val_dataset)
        
        # Test efficient augmentation system
        print("Testing efficient augmentation system:")
        print(f"  Base images in training: {len(train_dataset.images)}")
        print(f"  Augmentation strategies: {len(train_dataset.augmenters)}")
        print(f"  Total training samples: {len(train_dataset)}")
        
        # Show augmentation info
        print("\nAvailable augmentations:")
        aug_info = get_augmentation_info()
        for name, desc in aug_info.items():
            if name in train_dataset.augmenters:
                print(f"  ✓ {name}: {desc}")
        
        # Test sample loading
        print(f"\nTesting sample loading:")
        sample = train_dataset[0]
        print(f"  Sample 0 - Image shape: {sample['image'].shape}")
        print(f"  Sample 0 - Label shape: {sample['label'].shape}")
        print(f"  Sample 0 - Augmentation: {sample['aug_type']}")
        print(f"  Sample 0 - Base index: {sample['base_idx']}")
        
        # Test different augmentations of the same base image
        print(f"\nTesting augmentation variety:")
        base_idx = 0
        num_augs = len(train_dataset.augmenters)
        for i in range(min(3, num_augs)):
            sample = train_dataset[base_idx * num_augs + i]
            print(f"  Sample {base_idx * num_augs + i}: {sample['aug_type']} (base: {sample['base_idx']})")
        
        # Test dataloader
        print(f"\nTesting dataloader:")
        for i, batch in enumerate(train_loader):
            print(f"  Batch {i}: Images {batch['image'].shape}, Labels {batch['label'].shape}")
            if i >= 2:  # Only show first 3 batches
                break
        
        print(f"\n✅ All tests passed! Efficient augmentation system working correctly.")
        print(f"📊 Training efficiency: {len(train_dataset)} samples from {len(train_dataset.images)} base images")
        
        # Uncomment to visualize samples (requires matplotlib display)
        # print(f"\nVisualizing sample...")
        # visualize_cityscapes_sample(sample, show_plot=False)
        
    except FileNotFoundError as e:
        print(f"\n❌ Dataset not found: {e}")
        print(f"Please ensure Cityscapes dataset is available at the specified path.")
        print(f"Expected structure:")
        print(f"  {config.root_dir}/")
        print(f"    ├── leftImg8bit/")
        print(f"    │   ├── train/")
        print(f"    │   └── val/")
        print(f"    └── gtFine/")
        print(f"        ├── train/")
        print(f"        └── val/")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()