import abc
from typing import List, Any

from torch import nn
import torchmetrics
import torch

import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

import cfg
from models.networks import ModelBase


class ActiveLearnerBase(pl.LightningModule, abc.ABC):
    @abc.abstractmethod
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_network(self) -> None:
        raise NotImplementedError()

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward_mc_dropout(self, x: torch.Tensor, n_times: int, training: bool = False) -> torch.Tensor:
        raise NotImplementedError()

    def set_n_samples(self, n_samples: int) -> None:
        self.n_samples = n_samples

class ClassificationActiveLearner(ActiveLearnerBase):

    def __init__(self, net: ModelBase, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(num_classes=self.n_classes)
        self.val_accuracy   = torchmetrics.classification.Accuracy(num_classes=self.n_classes)
        self.test_accuracy  = torchmetrics.classification.Accuracy(num_classes=self.n_classes)

        self.val_confusion_matrix = torchmetrics.classification.ConfusionMatrix(num_classes=self.n_classes, normalize='all')
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_encoder(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.train_accuracy( torch.nn.Softmax()(preds), y)
        self.log(f'{self.n_samples}/Train_acc_step', self.train_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f'{self.n_samples}/Train_acc_epoch', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        self.log(f'{self.n_samples}/Train_loss_step', loss, on_step=True, on_epoch=False)
        self.log(f'{self.n_samples}/Train_loss_epoch', loss, on_step=False, on_epoch=True)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{self.n_samples}/{stage}_loss", loss, prog_bar=True)
            self.log(f"{self.n_samples}/{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'Val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'Test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def get_type(self) -> str:
        return 'Random'

    def forward_mc_dropout(self, x: torch.Tensor, n_times: int, training: bool = False) -> torch.Tensor:
        return self.net.forward_mc_dropout(x, n_times, training)

    def reset_network(self) -> None:
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.net.apply(fn=weight_reset)


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class UncertaintyClassificationActiveLearner(ClassificationActiveLearner):
    def __init__(self, net: ModelBase, n_classes: int, label_loader: DataLoader):
        super().__init__(net, n_classes)
        self.label_loader = label_loader
        self.bceloss = BCEWithLogitsLoss()

    def get_data(self):
        for (x, y), labels in self.label_loader:
            yield (x, y), labels

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_discriminator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.train_accuracy( torch.nn.Softmax()(preds), y)
        self.log(f'{self.n_samples}/Train_acc_step', self.train_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f'{self.n_samples}/Train_acc_epoch', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        self.log(f'{self.n_samples}/Train_loss_step', loss, on_step=True, on_epoch=False)
        self.log(f'{self.n_samples}/Train_loss_epoch', loss, on_step=False, on_epoch=True)

        sum = 0
        for i in range(64):
            (x, _), y = (next(self.get_data()))

            if y == 0:
                continue

            x = x.to(cfg.gpu)
            outs = self.net.forward_mc_dropout(x, 64, training=True)

            outs = torch.softmax(outs, dim=1)
            mean = torch.mean(outs, dim=0)
            H = -torch.sum(mean * torch.log(mean + 1e-10))
            E_H = -torch.mean(torch.sum(outs * torch.log(outs + 1e-10), dim=1), dim=0)
            negative_entropy = 0.01 * -(H - E_H)

            sum += negative_entropy

        return loss + sum



class DiscriminativeClassificationActiveLearner(ClassificationActiveLearner):
    def __init__(self, net: ModelBase, n_classes: int, label_loader: DataLoader):
        super().__init__(net, n_classes)
        self.label_loader = label_loader
        self.bceloss = BCEWithLogitsLoss()

    def get_data(self):
        for (x, y), labels in self.label_loader:
            yield (x, y), labels

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_discriminator(x)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        (x, _), y = (next(self.get_data()))
        x = x.to(cfg.gpu)
        y = y.to(cfg.gpu).unsqueeze(-1)
        logits = self.forward_discriminator(self.forward_encoder(x))
        loss_disc = 2 * self.bceloss(logits, y)
        loss += loss_disc

        return loss
