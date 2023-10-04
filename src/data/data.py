from typing import Tuple, Any

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST


class CIFAR10Album(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = np.array(img)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, target


class MNISTAlbum(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
               Args:
                   index (int): Index

               Returns:
                   tuple: (image, target) where target is index of the target class.
               """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        img = np.array(img).astype(np.float32) / 255.
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, target
