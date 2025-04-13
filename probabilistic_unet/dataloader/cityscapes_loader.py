import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


def prepare_aug_funcs(img_size):
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
    def __init__(self, root_dir, img_size=(512, 1024), split='train', mode='fine', target_type='semantic'):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.target_type = target_type
        self.img_size = img_size

        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, f'gt{mode.capitalize()}', split)

        self.images = []
        self.targets = []
        self.augmenters = prepare_aug_funcs(self.img_size)

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_dir, file_name)
                    if target_type == 'semantic':
                        target_suffix = '_gtFine_labelIds.png'
                    elif target_type == 'instance':
                        target_suffix = '_gtFine_instanceIds.png'
                    else:
                        raise ValueError(f"Unsupported target_type: {target_type}")
                    target_name = file_name.replace('_leftImg8bit.png', target_suffix)
                    target_path = os.path.join(target_dir, target_name)
                    self.images.append(img_path)
                    self.targets.append(target_path)

        seed = 200
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        return len(self.images) * len(self.augmenters)


    def __getitem__(self, idx):
        img_idx = idx // len(self.augmenters)
        aug_key = list(self.augmenters.keys())[idx % len(self.augmenters)]
        aug = self.augmenters[aug_key]

        img = Image.open(self.images[img_idx]).convert("RGB")
        mask = Image.open(self.targets[img_idx]).convert("L")


        img = aug["image"](img)
        mask = aug["mask"](mask)

        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()

        return {
            "image": img_tensor.float(),
            "label": mask_tensor.float(),
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CityscapesDataset(
        root_dir="../../datasets/Cityscapes",
        img_size=(512, 1024),
        split='train',
        mode='fine',
        target_type='semantic'
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    print("Dataset size:", len(dataset))

    print(dataset[0]["image"].shape)
    print(dataset[0]["label"].shape)

    # for i, sample in enumerate(dataloader):
    #     print(f"Batch {i} | Image Shape: {sample['image'].shape} | Label Shape: {sample['label'].shape}")

    #     # Visualize first batch only
    #     if i == 0:
    #         visualize_sample({
    #             "image": sample["image"][0],
    #             "label": sample["label"][0]
    #         })
    #         break