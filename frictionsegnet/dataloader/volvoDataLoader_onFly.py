import glob
import json
from frictionsegnet.dataloader import GenericDataLoader, volvoData
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

    def __init__(self, **kwargs):

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

        self.transform_in = preprocess_in = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Resize(self.imgSize),
            ]
        )
        self.transform_ou = preprocess_out = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Resize(self.imgSize)
            ]
        )

    def __len__(self):
        return len(self.imgNamesList) * len(self.augmenters.keys())

    def get_num_classes(self):
        if self.reducedCategories:
            return len(self.reducedCategoriesColors)
        else:
            return len(self.labels)

    def __getitem__(self, idx):

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

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            org_img = augmenterFunc(org_img)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seg_color = augmenterFunc(seg_color)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            seg_mask = augmenterFunc(seg_mask)

        elif augKey == "erasing" or augKey == "augMix":

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            org_img = augmenterFunc(org_img)

        else:
            org_img = augmenterFunc(org_img)

        return org_img, seg_mask, seg_color
