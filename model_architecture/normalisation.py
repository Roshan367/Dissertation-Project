import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np

"""
Layer normalisation module for the transformer decoder block
Normalises the input across the feature dimension using learnable
scale (gamma) and shift (beta) parameters
"""
class HybridLayerNorm(nn.Module):
    """
    Initialises learnable gamma (scale) and beta (shift) parameters,
    both of shape (d_model,), and sets the numerical stability epsilon
    """
    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon
        
    """
    Applies custom layer normalisation to the input tensor
    """
    def forward(self, x):
        return NumpyNormalisation.apply(x, self.weights, self.bias, self.epsilon)


"""
Custom layer normalisation with manual forward and backward passes
Implements layer normalisation from scratch using torch.autograd.Function,
allowing full control over gradient computation during backpropagation
"""
class NumpyNormalisation(torch.autograd.Function):
    """
    Forward pass for layer normalisation
    Computes mean and variance across the feature dimension, normalises
    the input, then applies the learnable affine transformation
    """
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        std = torch.sqrt(var + epsilon)
        xhat = (x - mean) / std

        y = gamma * xhat + beta

        ctx.save_for_backward(
            x,
            gamma,
            beta,
        )
        ctx.xhat = xhat
        ctx.std = std
        ctx.epsilon = epsilon
        return y
        
    """
    Manual backward pass for layer normalisation
    Computes gradients for the input x, gamma, and beta using the
    chain rule applied through the normalisation and affine transformation
    Returns None for epsilon as it is not a learnable parameter
    """
    @staticmethod
    def backward(ctx, dY):
        x, gamma, beta = ctx.saved_tensors
        xhat = ctx.xhat
        std = ctx.std
        epsilon = ctx.epsilon

        B, T, D = dY.shape

        dbeta = torch.sum(dY, dim=(0, 1))

        dgamma = torch.sum(dY * xhat, dim=(0, 1))

        dxhat = dY * gamma
        dstd = torch.sum(dxhat * (xhat * -1) / (std), dim=-1, keepdim=True)
        dvar = torch.sum(dxhat * (xhat * -0.5) / (std**2), dim=-1, keepdim=True)
        ddiff = dvar * 2 * xhat * std / (D) + dxhat / (std)
        dmean = torch.sum(dxhat * -1 / (std), dim=-1, keepdim=True)
        dx = ddiff + dmean / D

        return (
            dx,
            dgamma,
            dbeta,
            None,
        )
