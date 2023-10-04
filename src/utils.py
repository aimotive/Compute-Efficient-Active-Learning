import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor
from torch.autograd import Function
from torch.nn import Module
from torchvision.datasets import Cityscapes


def get_gpu() -> int:
    gpu = os.getenv('CUDA_VISIBLE_DEVICES')
    return int(gpu)


def output_to_img(output: torch.Tensor) -> torch.Tensor:
    def r(arr):
        return Cityscapes.classes[arr].color[0]

    def g(arr):
        return Cityscapes.classes[arr].color[1]

    def b(arr):
        return Cityscapes.classes[arr].color[2]

    output_np = output.cpu().detach().numpy()
    # Cityscapes.classes[]
    output_rgb = np.dstack([np.vectorize(r)(output_np), np.vectorize(g)(output_np), np.vectorize(b)(output_np)]) / 255.

    return torch.Tensor(output_rgb.transpose(2, 0, 1)).to(output.device)


class RevGrad(Function):

    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class RevGradModule(Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)
