import os
from datetime import  datetime
from typing import  List, Any, Dict
import time

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers, Trainer

import cfg
from data.data_factory import DataFactory
from models.model_base import ActiveLearnerBase
from sampling_strategies.strategies import SamplerBase


class ActiveLearningTrainer:
    def __init__(self,
                 data_factory: DataFactory,
                 network: ActiveLearnerBase,
                 sampler: SamplerBase,
                 batch_size: int = 32,
                 n_trainings: int = 10,
                 n_trainings_per_dataset: int = 1,
                 use_best: bool = True
                 ):

        self.use_best = use_best
        self.data_factory = data_factory
        self.net = network
        self.batch_size = batch_size
        self.n_trainings = n_trainings
        self.n_training_per_dataset = n_trainings_per_dataset
        self.sampler = sampler
        self.start_time = time.time()
        self.logger = self._create_logger()
        self.test_loader = self.data_factory.get_test_loader()

    def train(self) -> None:
        self.logger.experiment.config["type"]    = self.sampler.get_type()
        self.logger.experiment.config["dataset"] = self.data_factory.dataset_type
        self.logger.experiment.config["subsample"] = self.sampler.subsample_unlabeled
        self.logger.experiment.config["subsample_size"] = self.sampler.candidate_pool_size
        self.logger.experiment.config["use_best"] = self.use_best

        for _ in range(self.n_trainings):
            test_metrics_list = list()
            test_paths = list()

            # Train n times for better estimate of loss at given dataset
            for i in range(self.n_training_per_dataset):
                self.net.reset_network()

                if hasattr(self.net, 'label_loader'):
                    label_loader = self.data_factory.get_index_loader_from_indices(self.sampler.labeled_pool)
                    self.net.label_loader = label_loader

                self.net.set_n_samples(len(self.sampler.labeled_pool))

                test_metrics, test_path = self._train_test_net()
                test_paths.append(test_path)
                test_metrics_list.append(test_metrics)

            if self.use_best:
                best_path = self._get_best_path(test_metrics_list, test_paths)
                self.net = self.net.load_from_checkpoint(best_path)

            # Log results and update dataset
            test_metrics_avg = self._avg_dict(test_metrics_list)
            self._log_test_metrics(test_metrics_avg)

            unlabeled_loader = self.data_factory.get_unlabeled_loader_from_indices(self.sampler.get_unlabeled_pool())
            self.sampler.choose_best_labels_and_update(self.net, unlabeled_loader)

    @staticmethod
    def _avg_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        d_avg = dict()
        for d in dicts:
            for key, val in d.items():
                if key not in d_avg:
                    d_avg[key] = ( val / len(dicts) )
                else:
                    d_avg[key] += ( val / len(dicts) )

        return d_avg

    def _train_test_net(self) -> Any:
        train_loader, val_loader = self.data_factory.get_train_loaders_from_indices(self.sampler.labeled_pool,
                                                                                    self.sampler.validation_pool)

        early_stopping = EarlyStopping(monitor=f'{self.net.n_samples}/Val_acc', patience=15, mode='max')

        checkpoint_callback = ModelCheckpoint(
            monitor=f"{self.net.n_samples}/Val_acc",
            dirpath=f"saved_models/{self.sampler.get_type()}/{self.net.n_samples}",
            filename="best",
            save_top_k=1,
            mode="max",
        )

        trainer = Trainer(gpus=[cfg.gpu], logger=self.logger, callbacks=[early_stopping, checkpoint_callback],
                          log_every_n_steps=25, check_val_every_n_epoch=1, max_epochs=250)

        trainer.fit(self.net, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self.net = self.net.load_from_checkpoint(checkpoint_callback.best_model_path)

        if not self.use_best:
            os.remove(checkpoint_callback.best_model_path)

        self.net.n_samples = len(self.sampler.labeled_pool)
        metrics = trainer.test(self.net, self.test_loader)[0]
        return metrics, checkpoint_callback.best_model_path

    def _create_logger(self) -> pl_loggers.WandbLogger:
        strategy  = self.sampler.get_type()
        data_type = self.data_factory.dataset_type

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d-%H:%M")
        return pl_loggers.WandbLogger(name=f'{data_type}/{strategy}/{date_str}', project='active-learning')

    def _log_test_metrics(self, test_metrics: Dict[str, float], suffix: str = '') -> None:
        test_metric_dict = dict()

        for key, val in test_metrics.items():
            metric_type = key.split('/')[-1]
            test_metric_dict[metric_type + suffix] = val

        self.logger.log_metrics(test_metric_dict, len(self.sampler.labeled_pool))
        self.logger.log_metrics({'elapsed_time': ( time.time() - self.start_time ) / 60.0}, len(self.sampler.labeled_pool))

    def _log_metrics(self, test_metrics: Dict[str, float], suffix: str = '', i: int = 0) -> None:
        test_metric_dict = dict()

        for key, val in test_metrics.items():
            metric_type = key.split('/')[-1]
            test_metric_dict[metric_type + suffix] = val

        self.logger.log_metrics(test_metric_dict, i)
        self.logger.log_metrics({'elapsed_time': ( time.time() - self.start_time ) / 60.0}, len(self.sampler.labeled_pool))

    def _get_best_path(self, test_metrics_list, test_paths):
        accuracies = list()

        for metrics in test_metrics_list:
            for key, val in metrics.items():
                if 'acc' in key:
                    accuracies.append(metrics[key])

        id = np.argmax(accuracies)
        return test_paths[id]


