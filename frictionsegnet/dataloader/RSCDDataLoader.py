from frictionsegnet.dataloader import GenericDataLoader, RSCDClassNames
from pathlib import Path
import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class RSCDDataLoader(GenericDataLoader):
    """A DataLoader for Remote Sensing Change Detection (RSCD) dataset.

    This class extends SuperDataLoader to handle loading and preprocessing of
    RSCD image data. It supports both training and validation modes, and handles
    image transformations and mask generation for semantic segmentation tasks.

    Args:
        **kwargs: Keyword arguments including:
            RSCD_cat (list): List of RSCD categories to load
            RSCDRootPath (str): Root path to the RSCD dataset
            mode (str): Either 'train' or 'val'
            imgSize (tuple): Target size for image resizing
            reducedCategories (bool): Whether to use reduced category set
            reducedCategoriesColors (dict): Mapping of categories to colors

    Attributes:
        RSCD_cat (list): List of selected RSCD categories
        datasetRootPath (Path): Path object pointing to dataset root
        RSCDClassNames (dict): Mapping of class names to labels
        dataset (np.ndarray): Array of image paths
        transform_in (transforms.Compose): Input image transformations
        transform_ou (transforms.Compose): Output mask transformations
    """

    def __init__(self, **kwargs):
        """Initialize the RSCDDataLoader with given parameters."""
        super().__init__(**kwargs)

        self.RSCD_cat = kwargs["RSCD_cat"]
        self.datasetRootPath = Path(kwargs["RSCDRootPath"])
        self.RSCDClassNames = RSCDClassNames

        tmpDataset = []

        # Load image paths based on mode
        if self.mode == "val":
            tmpDataset.extend(
                [
                    path
                    for path in glob.glob(
                        str(Path.joinpath(self.datasetRootPath, self.mode, "*.jpg"))
                    )
                    if any(
                        [
                            cat.replace("_", "-")
                            in "-".join(
                                path.split(os.sep)[-1].split(".")[0].split("-")[1:]
                            ).replace("_", "-")
                            for cat in self.RSCD_cat
                        ]
                    )
                ]
            )

        elif self.mode == "train":
            for cat in self.RSCD_cat:
                tmpDataset.extend(
                    glob.glob(
                        str(
                            Path.joinpath(self.datasetRootPath, self.mode, cat, "*.jpg")
                        )
                    )
                )

        self.dataset = np.array(tmpDataset)

        # Define input image transformations
        self.transform_in = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4191, 0.4586, 0.4700], [0.2553, 0.2675, 0.2945]
                ),
                transforms.Resize(self.imgSize),
            ]
        )

        # Define output mask transformations
        self.transform_ou = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(self.imgSize)]
        )

    def get_num_classes(self):
        """Get the number of classes in the dataset.

        Returns:
            int: Number of classes based on configuration
        """
        if self.reducedCategories:
            return len(self.reducedCategoriesColors)
        return len(self.labels)

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.dataset[:])

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing:
                'image': Transformed input image tensor
                'label': Binary label tensor
                'seg': Segmentation color mask tensor
                'FriLabel': Friction label tensor
        """
        # Load and process image
        img = Image.open(self.dataset[idx])

        # Extract category from filename
        cat = self.RSCDClassNames[
            "-".join(self.dataset[idx].split(os.sep)[-1].split(".")[0].split("-")[1:])
        ]

        # Create segmentation mask
        seg_mask = np.full_like(
            np.array(img), list(self.reducedCategoriesColors.keys()).index(cat)
        )

        # Generate probability masks
        label, seg_color, fricLabel = self.create_prob_mask_patches(seg_mask[:, :, 0])

        # Apply transformations
        if self.transform_in:
            img = self.transform_in(img)
            seg_color = transforms.Resize((256, 256))(transforms.ToTensor()(seg_color))
        if self.transform_ou:
            label = self.transform_ou(label)
            fricLabel = self.transform_ou(fricLabel)

        return {
            "image": img.type(torch.float),
            "label": label.type(torch.float),
            "seg": seg_color.type(torch.float),
            "FriLabel": fricLabel.type(torch.float),
        }
