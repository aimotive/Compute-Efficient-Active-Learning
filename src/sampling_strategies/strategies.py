import abc
import sys
from typing import List, Sequence, Union

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

import cfg
from data.data_factory import DataFactory
from models.model_base import ActiveLearnerBase
from utils import get_gpu
from scipy.stats import entropy, stats


class SamplerBase(abc.ABC):
    def __init__(self,
                 all_indices: Sequence[int],
                 start_ratio: Union[float, int],
                 validation_size: int,
                 acquisition_size: int,
                 data_factory: DataFactory,
                 subsample_unlabeled: bool,
                 candidate_pool_size: int = 10_000,
                 top_n: int = 0):

        self.top_n = top_n
        self.candidate_pool_size = candidate_pool_size
        self.all_indices = all_indices
        self.start_ratio = start_ratio
        self.validation_size = validation_size
        self.acquisition_size = acquisition_size
        self.data_factory = data_factory
        self.subsample_unlabeled = subsample_unlabeled

        self.labeled_pool, self.unlabeled_pool, self.validation_pool = list(), all_indices, list()
        self._init_starting_indices()
        self.candidate_pool = self.unlabeled_pool.copy()
        self.unlabeled_pool_values = np.ones_like(self.unlabeled_pool, dtype=np.float)

    def _init_starting_indices(self) -> None:
        assert self.start_ratio > 0
        assert self.validation_size > 0

        if self.start_ratio > 1.:
            self.start_ratio = int(self.start_ratio)
        else:
            self.start_ratio = int(len(self.all_indices) * self.start_ratio)

        if self.validation_size > 1.:
            self.validation_size = int(self.validation_size)
        else:
            self.validation_size = int(len(self.all_indices) * self.validation_size)

        self.labeled_pool = self.get_n_balanced_from_unlabeled(n=self.start_ratio)
        self.unlabeled_pool = self.unlabeled_pool[np.where(np.isin(self.unlabeled_pool, self.labeled_pool) == False)[0]]

        self.validation_pool = self.get_n_balanced_from_unlabeled(n=self.validation_size)
        self.unlabeled_pool = self.unlabeled_pool[np.where(np.isin(self.unlabeled_pool, self.validation_pool) == False)[0]]

    def update_pool(self, new_labeled_indices: Sequence[int]) -> None:
        self.labeled_pool = np.concatenate([self.labeled_pool, new_labeled_indices])
        self.unlabeled_pool_values = self.unlabeled_pool_values[np.where(np.isin(self.unlabeled_pool, self.labeled_pool) == False)[0]]
        self.unlabeled_pool = self.unlabeled_pool[np.where(np.isin(self.unlabeled_pool, self.labeled_pool) == False)[0]]

    @abc.abstractmethod
    def choose_best_labels(self,
                           net: ActiveLearnerBase,
                           unlabeled_loader: DataLoader) -> Sequence[int]:
        raise NotImplementedError()

    def get_unlabeled_pool(self):
        if self.subsample_unlabeled and self.candidate_pool is not None:
            return self.candidate_pool
        else:
            return self.unlabeled_pool

    def choose_best_labels_and_update(self,
                                      net: ActiveLearnerBase,
                                      unlabeled_loader: DataLoader,
                                      update: bool = True) -> None:
        indices_to_label = self.choose_best_labels(net, unlabeled_loader)
        if self.subsample_unlabeled and self.candidate_pool is not None:
            true_indices = self.candidate_pool[indices_to_label]
            values = self._get_unlabeled_values()
            self.unlabeled_pool_values[np.isin(self.unlabeled_pool, self.candidate_pool)] = values
        else:
            true_indices = self.unlabeled_pool[indices_to_label]

        if update:
            self.update_pool(true_indices)

            if self.subsample_unlabeled:
                self._update_candidate_pool()

    @abc.abstractmethod
    def _get_unlabeled_values(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_type(self) -> str:
        raise NotImplementedError()

    def get_n_balanced_from_unlabeled(self, n: int) -> Sequence[int]:
        assert n % 10 == 0, 'n should be divisible by number of classes!'
        assert n > 0, 'n should be a positive number!'

        sample_per_label = int(n / 10)
        data = self.data_factory.train_dataset_not_augmented
        labels = np.array([d[1] for d in data])
        labels = labels[self.unlabeled_pool]
        indices = [np.where(labels == i)[0] for i in range(10)]

        for index_list in indices:
            np.random.shuffle(index_list)

        indices = np.array([index_list[0: sample_per_label] for index_list in indices]).reshape(-1)
        indices = self.unlabeled_pool[indices]
        return indices

    @staticmethod
    def _enable_dropout(net: nn.Module) -> None:
        for m in net.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @staticmethod
    def _disable_dropout(net: nn.Module) -> None:
        for m in net.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.eval()

    def _update_candidate_pool(self):
        softmaxed = softmax(self.unlabeled_pool_values ** 1.2)
        pool_size = np.min([len(self.unlabeled_pool_values), self.candidate_pool_size])
        self.candidate_pool = np.random.choice(self.unlabeled_pool, size=pool_size, replace=False, p=softmaxed)


class RandomSampler(SamplerBase):
    def get_type(self) -> str:
        return 'Random'

    def _get_unlabeled_values(self):
        return self.values

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        self.values = np.ones(len(unlabeled_loader))
        return np.random.choice(len(unlabeled_loader), self.acquisition_size, replace=False)


class EntropySampler(SamplerBase):
    def __init__(self, all_indices: Sequence[int], start_ratio: Union[float, int], validation_size: int,
                 acquisition_size: int, data_factory: DataFactory, subsample_unlabeled: bool, subsample_size, top_n: int):
        super().__init__(all_indices, start_ratio, validation_size, acquisition_size,
                         data_factory, subsample_unlabeled, subsample_size, top_n)
        self.entropies = np.ones_like(self.all_indices)

    def _get_unlabeled_values(self):
        return self.entropies

    def get_type(self) -> str:
        return 'Entropy'

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        entropies = self.get_entropies(net, unlabeled_loader)
        self.entropies = entropies
        self.mean = entropies.mean()
        return np.argpartition(entropies, -self.acquisition_size)[-self.acquisition_size:]

    def get_entropies(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> np.array:
        net.net.to(cfg.gpu)
        net.net.eval()
        self._enable_dropout(net.net)
        uncertanties = list()

        for (x, _) in tqdm(unlabeled_loader):
            with torch.no_grad():
                x = x.to(cfg.gpu)
                out = torch.softmax(net.forward_mc_dropout(x, 64), dim=1)
                mean = torch.mean(out, dim=0).cpu().detach().numpy()
                _entropy = entropy(mean)
                uncertanties.append(_entropy)

        self._disable_dropout(net.net)
        return np.array(uncertanties)


class EntropyKMeansSampler(EntropySampler):
    def get_type(self) -> str:
        return 'EntropyKMeans'

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        entropies = self.get_entropies(net, unlabeled_loader)

        n_indices_to_cluster = self.acquisition_size * 5
        if n_indices_to_cluster > len(unlabeled_loader):
            n_indices_to_cluster = len(unlabeled_loader)

        indices_to_cluster = np.argpartition(entropies, -n_indices_to_cluster)[-n_indices_to_cluster:]
        entropies_to_cluster = entropies[indices_to_cluster]
        return self.get_best_clustered_indices(net, unlabeled_loader, entropies_to_cluster, indices_to_cluster)

    def get_best_clustered_indices(self, net: ActiveLearnerBase,
                                   unlabeled_loader: DataLoader,
                                   entropies_to_cluster: Sequence[float],
                                   indices_to_cluster: Sequence[int]) -> Sequence[int]:
        features_list = list()
        dataset = unlabeled_loader.dataset

        for idx in indices_to_cluster:
            (x, _) = dataset[idx]
            x = x.unsqueeze(0)
            x = x.to(cfg.gpu)

            features = net.forward_encoder(x).cpu().detach().numpy()

            features_list.append(features[0])

        kmeans = KMeans(n_clusters=self.acquisition_size, n_init=5, init='k-means++').fit(features_list)

        best_indices = []

        for i in range(self.acquisition_size):
            indices = np.where(kmeans.labels_ == i)[0]
            best_idx_by_cluster = indices_to_cluster[indices[np.argmax(entropies_to_cluster[indices])]]
            best_indices.append(best_idx_by_cluster)

        return np.array(best_indices)


class VarMaxSampler(SamplerBase):
    def _get_unlabeled_values(self):
        return self.varmaxes

    def get_type(self) -> str:
        return 'Varmax'

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        varmaxes = self.get_varmaxes(net, unlabeled_loader)
        self.varmaxes = varmaxes
        return np.argpartition(varmaxes, -self.acquisition_size)[-self.acquisition_size:]

    def get_varmaxes(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> np.array:
        net.net.to(cfg.gpu)
        net.net.eval()
        self._enable_dropout(net.net)
        varmaxes = list()

        for (x, _) in tqdm(unlabeled_loader):
            with torch.no_grad():
                x = x.to(cfg.gpu)
                x_input = x.repeat(128, 1, 1, 1)
                out = torch.nn.Softmax()(net(x_input))
                mean = torch.mean(out, dim=0).cpu().detach().numpy()
                varmax = 1 - np.max(mean)
                varmaxes.append(varmax)

        self._disable_dropout(net.net)
        return np.array(varmaxes)


class DiscriminativeSampler(SamplerBase):
    def get_type(self) -> str:
        return 'Discriminative'

    def _get_unlabeled_values(self):
        raise NotImplementedError()

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        confidences = self.get_confidences(net, unlabeled_loader)
        return np.argpartition(confidences, -self.acquisition_size)[-self.acquisition_size:]

    def get_confidences(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        net.net.to(cfg.gpu)
        net.net.eval()
        confidences = list()

        for (x, _) in tqdm(unlabeled_loader):
            with torch.no_grad():
                x = x.to(cfg.gpu)
                logits = net.forward_discriminator(net.forward_encoder(x))
                out = logits
                confidences.append(out.cpu().detach().numpy()[0][0])

        return np.array(confidences)


class VariationRatioSampler(SamplerBase):
    def __init__(self, all_indices: Sequence[int], start_ratio: Union[float, int], validation_size: int,
                 acquisition_size: int, data_factory: DataFactory, subsample_unlabeled: bool, subsample_size, top_n: int):
        super().__init__(all_indices, start_ratio, validation_size, acquisition_size,
                         data_factory, subsample_unlabeled, subsample_size, top_n)
        self.var_ratios = np.ones_like(self.all_indices)

    def _get_unlabeled_values(self):
        return self.var_ratios

    def get_type(self) -> str:
        return 'Variation ratio'

    def choose_best_labels(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> Sequence[int]:
        var_ratios = self.get_var_ratios(net, unlabeled_loader)
        self.var_ratios = var_ratios
        return np.argpartition(var_ratios, -self.acquisition_size)[-self.acquisition_size:]

    def get_var_ratios(self, net: ActiveLearnerBase, unlabeled_loader: DataLoader) -> np.array:
        net.net.to(cfg.gpu)
        net.net.eval()
        self._enable_dropout(net.net)
        var_ratios = list()

        for (x, _) in tqdm(unlabeled_loader):
            with torch.no_grad():
                x = x.to(cfg.gpu)
                x_input = x.repeat(128, 1, 1, 1)
                out = torch.nn.Softmax()(net(x_input))
                y_pred = torch.argmax(out, dim=1)
                mode_count = torch.max(torch.bincount(y_pred))
                var_ratio = 1 - mode_count / 128
                var_ratios.append(var_ratio.cpu())

        self._disable_dropout(net.net)
        return np.array(var_ratios)


def get_sampler(strategy: str,
                all_indices: Sequence[int],
                start_ratio: float,
                validation_size: int,
                acquisition_size: int,
                data_factory: DataFactory,
                subsample_unlabeled: bool,
                subsample_size: int,
                top_n: int) -> SamplerBase:
    strategy = strategy.lower()
    if strategy == 'random':
        return RandomSampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    elif strategy == 'entropy':
        return EntropySampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    elif strategy == 'entropy_kmeans':
        return EntropyKMeansSampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    elif strategy == 'varmax':
        return VarMaxSampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    elif strategy == 'discriminative':
        return DiscriminativeSampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    elif strategy == 'var_ratio':
        return VariationRatioSampler(all_indices, start_ratio, validation_size, acquisition_size, data_factory, subsample_unlabeled, subsample_size, top_n)
    else:
        raise NotImplementedError(f'Sampling method {strategy} not implemented')


