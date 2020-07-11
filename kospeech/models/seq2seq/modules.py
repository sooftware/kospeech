import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by He initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LayerNorm(nn.Module):
    """ Wrapper class of torch.nn.LayerNorm """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        output = (z - mean) / (std + self.eps)
        output = self.gamma * output + self.beta

        return output


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential modeule. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs):
        if self.contiguous:
            inputs = inputs.contiguous()

        return inputs.view(*self.shape)
