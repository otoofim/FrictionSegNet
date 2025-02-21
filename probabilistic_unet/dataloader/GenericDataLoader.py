from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from pathlib import Path
from os.path import exists, join
from PIL import ImageFile, Image
from cityscapesscripts.helpers.labels import labels as city_labels
from torchvision.datasets import Cityscapes
from torchvision import transforms
from MapillaryIntendedObjs import classIds, friClass
import glob
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GenericDataLoader(Dataset):
    """
    A custom dataset loader for semantic segmentation tasks that supports both Cityscapes and Mapillary datasets.
    This loader handles image loading, segmentation mask processing, and friction label generation.

    Features:
    - Supports both training and validation modes
    - Handles reduced category mapping for simplified segmentation tasks
    - Generates probability masks for segmentation
    - Creates friction labels for each segmentation class
    - Supports conversion between segmentation masks and color representations

    Args:
        **kwargs: Keyword arguments including:
            mode (str): Dataset mode - either 'train' or 'val'
            input_img_dim (tuple): Input image dimensions (height, width)
            reducedCategories (bool): Whether to use reduced category mapping

    Attributes:
        mode (str): Current dataset mode
        imgSize (tuple): Input image dimensions
        reducedCategoriesColors (dict): Mapping of reduced categories to colors
        friClass (dict): Mapping of classes to friction values
        reducedCategories (bool): Flag for using reduced categories
    """

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs["mode"] not in ["train", "val"]:
            raise ValueError("Valid values for mode argument are: train, val")

        self.mode = kwargs["mode"]
        self.imgSize = kwargs["input_img_dim"]
        self.reducedCategoriesColors = classIds
        self.friClass = friClass
        self.reducedCategories = kwargs["reducedCategories"]

        self.pixel_to_color = np.vectorize(self.return_color)

    def create_prob_mask(self, seg_mask, seg_color):
        """
        Creates a probability mask and friction label from a segmentation mask.

        Args:
            seg_mask (np.ndarray): Input segmentation mask
            seg_color (np.ndarray): Color representation of segmentation mask

        Returns:
            tuple: (probability_mask, colored_segmentation, friction_label)
                - probability_mask: One-hot encoded segmentation mask
                - colored_segmentation: Color-coded segmentation visualization
                - friction_label: Corresponding friction values for each pixel
        """
        fricLabel = torch.zeros(seg_mask.shape)

        if self.reducedCategories:
            tmpSeg = np.zeros(seg_mask.shape)
            for i, label in enumerate(self.labels):
                classid = list(self.reducedCategoriesColors.keys())[
                    list(self.reducedCategoriesColors.values()).index(
                        self.NewColors[label]
                    )
                ]
                seg_color[np.all(seg_color == self.labels[label], axis=-1)] = (
                    self.reducedCategoriesColors[classid]
                )
                tmpSeg[seg_mask == i] = list(self.reducedCategoriesColors.keys()).index(
                    classid
                )

            seg_mask = tmpSeg.astype(int)

            # Create friction label
            for i, className in enumerate(self.reducedCategoriesColors):
                fricLabel = np.where(seg_mask == i, self.friClass[className], fricLabel)

        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        return label, seg_color, fricLabel

    def create_prob_mask_patches(self, seg_mask):
        """
        Creates probability masks for image patches with consistent class labels.

        Args:
            seg_mask (np.ndarray): Input segmentation mask for the patch

        Returns:
            tuple: (probability_mask, colored_segmentation, friction_label)
                - probability_mask: One-hot encoded segmentation mask
                - colored_segmentation: Color-coded visualization
                - friction_label: Friction values for the patch
        """
        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1
        fricLabel = np.ones(seg_mask.shape)
        fricLabel *= list(self.friClass.values())[seg_mask[0, 0]]

        seg_color = self.prMask_to_color(
            torch.tensor(label).permute(2, 0, 1).unsqueeze(0)
        )
        seg_color = seg_color.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        return label, seg_color, fricLabel

    def return_color(self, idx):
        """
        Converts a class index to its corresponding color tuple.

        Args:
            idx (int): Class index

        Returns:
            tuple: RGB color values for the class
        """
        if self.reducedCategories:
            return tuple(
                self.reducedCategoriesColors[
                    list(self.reducedCategoriesColors.keys())[int(idx)]
                ]
            )
        else:
            return tuple(self.labels[list(self.labels.keys())[int(idx)]])

    def prMask_to_color(self, img):
        """
        Converts a probability mask to a colored segmentation visualization.

        Args:
            img (torch.Tensor): Input probability mask

        Returns:
            torch.Tensor: Colored segmentation visualization
        """
        argmax = torch.argmax(img, dim=1)
        resu = self.pixel_to_color(argmax)
        return (
            torch.tensor(
                np.transpose(np.stack((resu[0], resu[1], resu[2])), (1, 0, 2, 3))
            ).float()
            / 255
        )

    def separateClasses(self, seg_mask):
        """
        Separates a segmentation mask into individual binary masks for each class.

        Args:
            seg_mask (np.ndarray): Input segmentation mask

        Returns:
            np.ndarray: Array of binary masks for each class
        """
        segMasks = []
        for classLabel in range(len(self.reducedCategoriesColors)):
            argmaxs = torch.argmax(torch.tensor(seg_mask), axis=-1)
            converted = torch.where(argmaxs == classLabel, argmaxs, 0)

            label = np.zeros(
                (
                    converted.shape[0],
                    converted.shape[1],
                    len(self.reducedCategoriesColors),
                )
            )
            indexs = np.ix_(
                np.arange(converted.shape[0]), np.arange(converted.shape[1])
            )
            label[indexs[0], indexs[1], converted] = 1
            segMasks.append(label)

        return np.array(segMasks)
