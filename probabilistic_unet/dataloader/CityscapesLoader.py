from probabilistic_unet.dataloader import GenericDataLoader
from pathlib import Path
from os.path import exists
import glob
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from torchvision import transforms
from cityscapesscripts.helpers.labels import labels as city_labels
from PIL import Image
import torch


class CityscapesLoader(GenericDataLoader):
    """A data loader for the Cityscapes dataset that inherits from GenericDataLoader.

    This class handles loading and preprocessing of Cityscapes dataset images and their
    corresponding segmentation masks. It supports both full and reduced category sets
    and provides necessary transformations for both input images and labels.

    Args:
        **kwargs: Keyword arguments including:
            cityscapesRootPath (str): Root directory path for Cityscapes dataset
            mode (str): Dataset mode ('train', 'val', or 'test')
            imgSize (Tuple[int, int]): Target size for image resizing
            reducedCategories (bool): Whether to use reduced category set
            mapillaryNewColors (List[str]): List of Mapillary color categories
            reducedCategoriesColors (Optional[List[Tuple[int, int, int]]]): Colors for reduced categories

    Attributes:
        datasetRootPath (Path): Path object pointing to dataset root directory
        dataset (np.ndarray): Array of paths to valid image files
        labels (Dict): Mapping between Mapillary colors and Cityscapes colors
        pixel_to_color (np.vectorize): Vectorized function for color conversion
        transform_in (transforms.Compose): Input image transformation pipeline
        transform_ou (transforms.Compose): Output label transformation pipeline
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.datasetRootPath = Path(kwargs["cityscapesRootPath"])

        # Build dataset of valid image paths
        self.dataset = self._build_dataset()

        # Initialize color mappings
        self.labels = self._initialize_labels()

        # Create vectorized color conversion function
        self.pixel_to_color = np.vectorize(self.return_color)

        # Setup transformations
        self._setup_transformations()

    def _build_dataset(self) -> np.ndarray:
        """Builds dataset by collecting valid image paths.

        Returns:
            np.ndarray: Array of valid image file paths.
        """
        tmp_dataset = []
        image_pattern = str(
            Path.joinpath(self.datasetRootPath, "images", self.mode, "*.jpg")
        )

        for img_path in glob.glob(image_pattern):
            if (exists(str(img_path).replace("images", "color"))) and (
                exists(str(img_path).replace("images", "masks"))
            ):
                tmp_dataset.append(img_path)

        return np.array(tmp_dataset)

    def _initialize_labels(self) -> Dict:
        """Initializes label mappings between Mapillary and Cityscapes colors.

        Returns:
            Dict: Mapping between Mapillary color names and Cityscapes RGB values.
        """
        labels = {}
        tmp_labels = {
            label.name.replace(" ", "-"): label.color for label in city_labels
        }

        for mapillary_color in self.mapillaryNewColors:
            for cityscapes_label in tmp_labels:
                if cityscapes_label in mapillary_color:
                    labels[mapillary_color] = tmp_labels[cityscapes_label]

        return labels

    def _setup_transformations(self) -> None:
        """Sets up input and output transformation pipelines."""
        self.transform_in = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.2867, 0.3250, 0.2837], [0.1862, 0.1895, 0.1865]
                ),
                transforms.Resize(self.imgSize),
            ]
        )

        self.transform_ou = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.imgSize)]
        )

    def get_num_classes(self) -> int:
        """Returns the number of classes in the dataset.

        Returns:
            int: Number of classes (either reduced or full set).
        """
        return (
            len(self.reducedCategoriesColors)
            if self.reducedCategories
            else len(self.labels)
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Retrieves a sample from the dataset at the given index.

        Args:
            idx (Union[int, torch.Tensor]): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'image': Transformed input image
                - 'label': Transformed segmentation label
                - 'seg': Transformed segmentation color mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load images and masks
        img = Image.open(self.dataset[idx])
        seg_mask = np.array(Image.open(self.dataset[idx].replace("images", "masks")))
        seg_color = np.array(
            Image.open(self.dataset[idx].replace("images", "color")).convert("RGB")
        )

        # Create probability mask
        label, seg_color = self.create_prob_mask(seg_mask, seg_color)

        # Apply transformations
        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.ToTensor()(seg_color)
        if self.transform_ou:
            label = self.transform_ou(label)

        return {"image": img, "label": label, "seg": seg_color}
