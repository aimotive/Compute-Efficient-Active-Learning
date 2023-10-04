from pathlib import Path
from typing import Tuple, List, Union, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.dataset import T_co
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose

import cfg
from data.data import CIFAR10Album, MNISTAlbum


class IsLabeledDatasetWrapper:
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[index], np.array(index not in self.indices).astype(np.float32)


class DataFactory:
    def __init__(self,
                 dataset_type: str,
                 batch_size: int = 64,
                 transforms: Tuple[Compose, Compose] = (None, None),
                 validation_split: float = 0.1,
                 data_path: Path = None):

        self.dataset_type = dataset_type.lower()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_transform, self.test_transform = transforms
        self.validation_split = validation_split

        self.train_dataset_augmented = self._create_train_dataset(augment=True)
        self.train_dataset_not_augmented = self._create_train_dataset(augment=False)
        self.test_dataset = self._create_test_dataset()

    def get_test_loader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

        return loader

    def get_unlabeled_loader_from_indices(self, indices: List[int]) -> DataLoader:
        dataset = Subset(self.train_dataset_not_augmented, indices)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )

    # def get_train_loaders_from_indices(self, indices: List[int]) -> Tuple[DataLoader, DataLoader]:
    #     """
    #     Returns (train_loader, val_loader)
    #     """
    #     train_indices, val_indices = train_test_split(indices, test_size=self.validation_split)
    #
    #     train_sampler = SubsetRandomSampler(train_indices)
    #     val_sampler   = SubsetRandomSampler(val_indices)
    #
    #     train_loader = DataLoader(
    #                     self.train_dataset_augmented,
    #                     batch_size=self.batch_size,
    #                     sampler=train_sampler,
    #                     num_workers=4,
    #                     drop_last=False,
    #                     pin_memory=True
    #                 )
    #
    #     val_loader = DataLoader(
    #                     self.train_dataset_not_augmented,
    #                     batch_size=self.batch_size,
    #                     sampler=val_sampler,
    #                     num_workers=4,
    #                     drop_last=False,
    #                     pin_memory=True
    #                 )
    #
    #     return train_loader, val_loader

    def get_train_loaders_from_indices(self, train_indices: List[int], val_indices: List[int]) -> Tuple[DataLoader, DataLoader]:
        """
        Returns (train_loader, val_loader)
        """
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
                        self.train_dataset_augmented,
                        batch_size=self.batch_size,
                        sampler=train_sampler,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True
                    )

        val_loader = DataLoader(
                        self.train_dataset_not_augmented,
                        batch_size=self.batch_size,
                        sampler=val_sampler,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True
                    )

        return train_loader, val_loader

    def get_index_loader_from_indices(self, indices: Sequence[int]) -> DataLoader:
        wrapper = IsLabeledDatasetWrapper(self.train_dataset_not_augmented, indices)
        n_train = len(self.train_dataset_not_augmented)
        n_indices = len(indices)
        n_no_indices = n_train - n_indices
        weight_indices = n_train / float(n_indices)
        weight_no_indices = n_train / float(n_no_indices)

        weights = [weight_indices if idx in indices else weight_no_indices for idx in range(n_train)]

        sampler = WeightedRandomSampler(weights, n_train, replacement=True)

        loader = DataLoader(
            wrapper,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        return loader

    def get_train_length(self) -> int:
        return len(self.train_dataset_augmented)

    def _create_train_dataset(self, augment: bool = True):
        return self._create_dataset(augment=augment, train=True)

    def _create_test_dataset(self) -> Dataset:
        return self._create_dataset(augment=False, train=False)

    def _create_dataset(self, augment: bool = False, train: bool = True) -> Dataset:
        download = self.data_path is None
        transform = self.train_transform if train else self.test_transform

        if not augment:
            transform = self.test_transform

        if self.dataset_type == 'cifar10':
            if download:
                return CIFAR10Album(root='./cifar10/', train=train, transform=transform, download=True)
            else:
                return CIFAR10Album(root=str(self.data_path), train=train, transform=transform)
        elif self.dataset_type == 'mnist':
            if download:
                return MNISTAlbum(root='./mnist/', train=train, transform=transform, download=True)
            else:
                return MNISTAlbum(root=str(self.data_path), train=train, transform=transform)
        else:
            raise NotImplementedError(f'Dataset {self.dataset_type} not implemented!')
