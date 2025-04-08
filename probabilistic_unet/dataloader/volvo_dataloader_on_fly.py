import glob
import json
from probabilistic_unet.dataloader import GenericDataLoader, volvoData
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from os.path import join
import random


def prepareAugFuncs(imgSize):
    """Prepares a dictionary of image augmentation functions.

    Creates a set of image transformation functions using torchvision transforms
    for data augmentation during training. Each function applies different types
    of augmentation like cropping, color jittering, blurring etc.

    Args:
        imgSize (tuple): Target size for the transformed images as (height, width).

    Returns:
        dict: A dictionary mapping augmentation names to their corresponding
            transform functions. Available augmentations include:
            - org: Basic resizing
            - crop: Center cropping
            - grayScale: Grayscale conversion
            - colorJitter: Random color adjustments
            - gaussianBlur: Gaussian blurring
            - rotation: Random rotation
            - elastic: Elastic transformation
            - invert: Color inversion
            - solarize: Image solarization
            - augMix: AugMix augmentation
            - posterize: Color posterization
            - erasing: Random erasing
    """
    return {
        "org": T.Compose(
            [
                T.ToTensor(),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "crop": T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(size=500),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "grayScale": T.Compose(
            [
                T.ToTensor(),
                T.Grayscale(num_output_channels=3),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "colorJitter": T.Compose(
            [
                T.ToTensor(),
                T.ColorJitter(
                    brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=0.5
                ),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "gaussianBlur": T.Compose(
            [
                T.ToTensor(),
                T.GaussianBlur(kernel_size=(51, 51), sigma=(5, 5)),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "rotation": T.Compose(
            [
                T.ToTensor(),
                T.RandomRotation(degrees=(-30, 30)),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "elastic": T.Compose(
            [
                T.ToTensor(),
                T.ElasticTransform(alpha=500.0),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "invert": T.Compose(
            [
                T.ToTensor(),
                T.RandomInvert(p=1.0),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "solarize": T.Compose(
            [
                T.ToTensor(),
                T.RandomSolarize(threshold=0.05, p=1.0),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "augMix": T.Compose(
            [
                T.AugMix(severity=10, mixture_width=10),
                T.ToTensor(),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "posterize": T.Compose(
            [
                T.RandomPosterize(bits=2, p=1.0),
                T.ToTensor(),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
        "erasing": T.Compose(
            [
                T.ToTensor(),
                T.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                T.Resize(imgSize),
                T.ToPILImage(),
            ],
        ),
    }


class volvo_onFly(GenericDataLoader):
    """Custom data loader for Volvo dataset with on-the-fly augmentations.

    This class implements a PyTorch dataset for loading and preprocessing Volvo
    image data with segmentation masks. It supports various image augmentations
    that are applied during runtime.

    Args:
        **kwargs: Keyword arguments including:
            volvoRootPath (str): Root directory containing the dataset.
            mode (str): Operating mode ('train', 'val', or 'test').
            input_img_dim (tuple): Desired dimensions for input images.

    Attributes:
        datasetRootPath (str): Path to the dataset root directory.
        mode (str): Current operating mode.
        imgSize (tuple): Target image dimensions.
        imgNamesList (list): List of image file paths.
        augmenters (dict): Dictionary of augmentation functions.
        labels (dict): Mapping of category names to colors.
        NewColors (dict): Predefined color mappings from volvoData.
        transform_in (transforms.Compose): Input image transformation pipeline.
        transform_ou (transforms.Compose): Output mask transformation pipeline.
    """

    def __init__(self, **kwargs):
        """Initializes the volvo_onFly dataset."""
        super().__init__(**kwargs)

        self.datasetRootPath = kwargs["volvoRootPath"]
        self.mode = kwargs["mode"]
        self.imgSize = kwargs["input_img_dim"]

        self.imgNamesList = glob.glob(join(self.datasetRootPath, "*.jpg"))
        self.augmenters = prepareAugFuncs(self.imgSize)

        labels = {}
        with open(
            str(Path.joinpath(Path(self.datasetRootPath), "config.json"))
        ) as jsonfile:
            config = json.load(jsonfile)
            for cat in config["labels"].keys():
                labels[cat] = config["labels"][cat]["color"]
        self.labels = labels
        self.NewColors = volvoData

        self.transform_in = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Resize(self.imgSize),
            ]
        )
        self.transform_ou = transforms.Compose([transforms.Resize(self.imgSize)])

    def __len__(self):
        """Returns the total number of samples in the dataset.

        The total count is the product of number of images and number of
        augmentations, as each image will be augmented in multiple ways.

        Returns:
            int: Total number of samples.
        """
        return len(self.imgNamesList) * len(self.augmenters.keys())

    def get_num_classes(self):
        """Returns the number of classes in the dataset.

        Returns:
            int: Number of classes (either reduced categories or full label set).
        """
        if self.reducedCategories:
            return len(self.reducedCategoriesColors)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - image: Transformed input image tensor
                - label: Segmentation mask tensor
                - seg: Colored segmentation mask tensor
                - FriLabel: Friction label tensor
        """
        org_img, seg_mask, seg_color = self.generateSample(idx)

        seg_mask, seg_color, fricLabel = self.create_prob_mask(
            np.array(seg_mask)[:, :, 0] - 1, np.array(seg_color)
        )

        if self.transform_in:
            org_img = self.transform_in(org_img)
            seg_color = transforms.Resize(self.imgSize)(
                transforms.ToTensor()(seg_color)
            )
        if self.transform_ou:
            label = self.transform_ou(torch.tensor(seg_mask).permute(2, 0, 1))
            fricLabel = self.transform_ou(torch.tensor(fricLabel).unsqueeze(0))

        return {
            "image": org_img.type(torch.float),
            "label": label.type(torch.float),
            "seg": seg_color.type(torch.float),
            "FriLabel": fricLabel.type(torch.float),
        }

    def generateSample(self, idx):
        """Generates an augmented sample from the dataset.

        This method loads an image and its corresponding masks, then applies
        the appropriate augmentation based on the index.

        Args:
            idx (int): Index of the sample to generate.

        Returns:
            tuple: Contains:
                - PIL.Image: Augmented input image
                - PIL.Image: Augmented segmentation mask
                - PIL.Image: Augmented colored segmentation mask

        Note:
            The method uses consistent random seeds for transformations that
            require randomization to ensure consistent augmentation across
            image and mask pairs.
        """
        org_img_idx = idx // len(self.augmenters.keys())

        org_img = Image.open(self.imgNamesList[org_img_idx])
        seg_mask = Image.open(
            self.imgNamesList[org_img_idx]
            .replace("images", "masks")
            .replace(".jpg", "_watershed_mask.png")
        )
        seg_color = Image.open(
            self.imgNamesList[org_img_idx]
            .replace("images", "masks_color")
            .replace(".jpg", "_color_mask.png")
        )

        seed = np.random.randint(27)
        augKey = list(self.augmenters.keys())[idx % len(self.augmenters.keys())]
        augmenterFunc = self.augmenters[augKey]

        if augKey == "crop" or augKey == "rotation":
            # Apply consistent random transforms to all images
            for _ in range(4):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            org_img = augmenterFunc(org_img)
            seg_color = augmenterFunc(seg_color)
            seg_mask = augmenterFunc(seg_mask)

        elif augKey == "erasing" or augKey == "augMix":
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            org_img = augmenterFunc(org_img)

        else:
            org_img = augmenterFunc(org_img)

        return org_img, seg_mask, seg_color
