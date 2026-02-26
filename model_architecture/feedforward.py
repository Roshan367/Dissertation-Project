import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np

"""
Feedforward class to be used in the decoder

A feedforward neural network for the inputs
"""


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff))
        self.b1 = nn.Parameter(torch.randn(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model))
        self.b2 = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        return CustomFeedForward.apply(x, self.W1, self.b1, self.W2, self.b2)


"""
Custom forward and backpropagation for the feedforward neural
network
"""


class CustomFeedForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, b1, W2, b2):
        z1 = torch.matmul(x, W1) + b1
        a1 = torch.maximum(z1, torch.tensor(0.0, device=z1.device, dtype=z1.dtype))

        y = torch.matmul(a1, W2) + b2

        ctx.save_for_backward(
            x,
            W1,
            b1,
            a1,
            W2,
            b2,
        )
        ctx.z1 = z1

        return y

    @staticmethod
    def backward(ctx, dY):
        x, W1, b1, a1, W2, b2 = ctx.saved_tensors
        z1 = ctx.z1

        da1 = torch.matmul(dY, W2.transpose(0, 1))

        dW2 = torch.matmul(
            a1.reshape(-1, a1.shape[-1]).transpose(0, 1), dY.reshape(-1, dY.shape[-1])
        )

        db2 = dY.sum(dim=(0, 1))

        dz1 = da1 * (z1 > 0)

        dx = torch.matmul(dz1, W1.transpose(0, 1))

        dW1 = torch.matmul(
            x.reshape(-1, x.shape[-1]).transpose(0, 1), dz1.reshape(-1, dz1.shape[-1])
        )

        db1 = dz1.sum(dim=(0, 1))

        return (
            dx,
            dW1,
            db1,
            dW2,
            db2,
        )
