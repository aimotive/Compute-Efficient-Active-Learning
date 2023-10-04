from pathlib import Path
from typing import Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def create_loaders(cityscapes_root: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transforms = A.Compose([
        A.Normalize(),
        # A.RandomBrightnessContrast(),
        # A.Rotate(),
        A.RandomResizedCrop(512, 512, ratio=(1., 1.)),
        A.HorizontalFlip(),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    train_data = CityscapesAlbumentations(cityscapes_root, mode='fine', target_type='semantic',
                                          split='train', transforms=train_transforms)

    val_data = CityscapesAlbumentations(cityscapes_root, mode='fine', target_type='semantic',
                                        split='val', transforms=val_transforms)

    test_data = CityscapesAlbumentations(cityscapes_root, target_type='semantic',
                                         split='test', transforms=val_transforms)

    train_loader = DataLoader(
        train_data,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


class CityscapesAlbumentations(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        image  = np.array(image)
        target = np.array(target)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=target)
            image = augmented['image'].float()
            target = augmented['mask'].long()

        return image, target