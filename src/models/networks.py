import abc

import torch
import torchvision
from torch import nn

from utils import RevGradModule
import torch.nn.functional as F

def get_network(dataset_type: str, strategy: str) -> nn.Module:
    if dataset_type.lower() == 'cifar10':
        # return CIFAR10CNN()
        return VGGBNDrop(num_classes=10)
    elif dataset_type.lower() == 'mnist':
        return MNISTCNN()
    else:
        raise NotImplementedError(f'Network for {dataset_type} not implemented!')


def enable_dropout(net: nn.Module) -> None:
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def disable_dropout(net: nn.Module) -> None:
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ModelBase(abc.ABC):
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('No discriminator for this model!')

    def forward_mc_dropout(self, x: torch.Tensor, n_times: int, training: bool = False) -> torch.Tensor:
        raise NotImplementedError()

    def reset_network(self) -> None:
        raise NotImplementedError()


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.last_pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class VGGBNDrop(nn.Module, ModelBase):
    """
    https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua
    """

    def __init__(self, num_classes):
        super(VGGBNDrop, self).__init__()
        self.num_classes = num_classes
        self.reset_network()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.features = torchvision.models.vgg11_bn(pretrained=True).features

    def reset_network(self) -> None:
        self.features = torchvision.models.vgg11_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.revgrad = RevGradModule(alpha=0.3)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(self.avgpool(self.features(x)))
        x = self.classifier(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward_encoder(self, x) -> torch.Tensor:
        return self.flatten(self.avgpool(self.features(x)))

    def forward_mc_dropout(self, x: torch.Tensor, n_times: int, training: bool = False) -> torch.Tensor:
        if not training:
            self.eval()

        enable_dropout(self)
        features = self.forward_encoder(x)
        features_rep = features.repeat(n_times, 1)
        out = self.forward_head(features_rep)
        disable_dropout(self)

        self.train()

        return out

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.revgrad(x))


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MNISTCNN(nn.Module, ModelBase):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.discriminator = nn.Sequential(
            RevGradModule(alpha=0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_encoder(x))

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def forward_mc_dropout(self, x: torch.Tensor, n_times: int, training: bool = False) -> torch.Tensor:
        if not training:
            self.eval()

        enable_dropout(self)
        x_in = x.repeat(n_times, 1, 1, 1)
        out = self.forward(x_in)
        disable_dropout(self)
        self.train()

        return out

    def reset_network(self) -> None:
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.apply(fn=weight_reset)


class CIFAR10CNN(nn.Module, ModelBase):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),

        )
        # fully connected layer, output 10 classes
        self.out = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.discriminator = nn.Sequential(
            RevGradModule(alpha=0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_encoder(x))

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)

    def forward_discriminator(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
