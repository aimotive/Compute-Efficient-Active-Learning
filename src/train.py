import os

import numpy as np
import pytorch_lightning as pl
import torchmetrics.classification
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
import torch
from pytorch_lightning import Trainer
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import loggers as pl_loggers
import seaborn as sn

from data.cityscapes_albumentations import CityscapesAlbumentations, create_loaders
from utils import plot_output, output_to_img


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 34
        self.net = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=self.n_classes)
        self.activation = nn.Identity()
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.JaccardIndex(num_classes=self.n_classes)
        self.val_accuracy   = torchmetrics.JaccardIndex(num_classes=self.n_classes)
        self.val_confusion_matrix = torchmetrics.classification.ConfusionMatrix(num_classes=self.n_classes, normalize='all')

    def forward(self, x):
        return self.activation(self.net(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.train_accuracy( torch.nn.Softmax()(preds), y)
        self.log('Train/acc_step', self.train_accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log('Train/acc_epoch', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        self.log('Train/loss_step', loss, on_step=True, on_epoch=False)
        self.log('Train/loss_epoch', loss, on_step=False, on_epoch=True)

        if batch_idx % 30 == 0:
            target = output_to_img(y[0])
            output = output_to_img(torch.argmax(preds[0], 0))
            image = torch.concat([target, output], dim=2)
            self.logger.experiment.add_image('Train/input_target_output', image, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.val_accuracy(torch.nn.Softmax()(preds), y)
        # self.val_confusion_matrix(torch.nn.Softmax()(preds), y)

        self.log('Val/acc', self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('Val/loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            target = output_to_img(y[0])
            output = output_to_img(torch.argmax(preds[0], 0))
            image = torch.concat([target, output], dim=2)
            self.logger.experiment.add_image('Val/input_target_output', image, self.current_epoch)

        return loss

    def on_validation_end(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train():
    train_loader, val_loader, test_loader = create_loaders('/home/ad.adasworks.com/gabor.nemeth/work/datasets/cityscapes')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = LitModel()
    tb_logger = pl_loggers.TensorBoardLogger("logs/", name='testerino')
    early_stopping = EarlyStopping(monitor='Val/loss', patience=6, min_delta=0.001)

    checkpoint_callback = ModelCheckpoint(
        monitor="Val/loss",
        dirpath="models/",
        filename="cityscapes-deeplabv3+-{epoch:02d}-{Val/loss:.5f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(gpus=1, logger=tb_logger, log_every_n_steps=25, check_val_every_n_epoch=5,
                      callbacks=[early_stopping, checkpoint_callback],
                      max_epochs=99999)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.test(model, test_dataloaders=test_loader)


if __name__ == '__main__':
    train()

