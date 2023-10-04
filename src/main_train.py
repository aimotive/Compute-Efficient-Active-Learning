import argparse
from pathlib import Path
from typing import  Tuple

import numpy as np
import torch

import cfg
import albumentations as A
from active_learner import ActiveLearningTrainer
from albumentations.pytorch import ToTensorV2
from data.data_factory import DataFactory
from models.model_base import ClassificationActiveLearner, DiscriminativeClassificationActiveLearner, \
    UncertaintyClassificationActiveLearner
from models.networks import get_network
from sampling_strategies.strategies import get_sampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, choices=['cifar10', 'mnist'])
    parser.add_argument('--data-root', default=None, type=Path)
    parser.add_argument('--strategy', default='random', type=str, choices=['random', 'entropy', 'entropy_kmeans', 'discriminative', 'varmax'])
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--acquisition-size', default=500, type=int)
    parser.add_argument('--subsample-unlabeled', action='store_true')
    parser.add_argument('--subsample-size', default=10_000, type=int)
    parser.add_argument('--start-ratio', default=0.1, type=float)
    parser.add_argument('--validation-size', default=100, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=420, type=int)
    parser.add_argument('--n-trainings', default=10, type=int)
    parser.add_argument('--n-trainings-per-dataset', default=3, type=int)
    parser.add_argument('--repeat-experiment', default=1, type=int)
    parser.add_argument('--use-best', action='store_true')
    parser.add_argument('--use-top-n', default=0, type=int)
    parser.add_argument('--uncertainty-reg', action='store_true')

    return parser.parse_args()


def get_transforms(dataset_type: str) -> Tuple[A.Compose, A.Compose]:
    dataset_type = dataset_type.lower()

    if dataset_type in ['cifar10']:
        train_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    elif dataset_type == 'mnist':
        train_transforms = A.Compose([
            ToTensorV2()
        ])
        test_transforms = A.Compose([
            ToTensorV2()
        ])
    return test_transforms, train_transforms


def train(args, i: int):
    net = get_network(dataset_type=args.dataset, strategy=args.strategy)
    net.to(cfg.gpu)

    test_transforms, train_transforms = get_transforms(args.dataset)
    data_factory = DataFactory(args.dataset, args.batch_size, transforms=(train_transforms, test_transforms))

    np.random.seed(args.seed + i)
    torch.manual_seed(args.seed + i)

    sampler = get_sampler(args.strategy, np.arange(data_factory.get_train_length()),
                          args.start_ratio, args.validation_size,
                          args.acquisition_size, data_factory,
                          args.subsample_unlabeled, args.subsample_size,
                          args.use_top_n)
    label_loader = data_factory.get_index_loader_from_indices(sampler.labeled_pool)

    np.random.seed()
    torch.manual_seed(np.random.randint(0, 69420, 1))

    if args.strategy.lower() == 'discriminative':
        active_learner = DiscriminativeClassificationActiveLearner(net=net, n_classes=10, label_loader=label_loader)
    else:
        if args.uncertainty_reg:
            active_learner = UncertaintyClassificationActiveLearner(net=net, n_classes=10, label_loader=label_loader)
        else:
            active_learner = ClassificationActiveLearner(net=net, n_classes=10)

    learner = ActiveLearningTrainer(data_factory, active_learner, sampler, n_trainings=args.n_trainings,
                                    n_trainings_per_dataset=args.n_trainings_per_dataset,
                                    use_best=args.use_best)

    learner.train()


def main():
    args = parse_args()
    cfg.gpu = args.gpu

    for i in range(args.repeat_experiment):
        train(args, i)


if __name__ == '__main__':
    main()
