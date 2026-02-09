import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np


class HybridLayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        return NumpyNormalisation.apply(x, self.weights, self.bias, self.epsilon)


class NumpyNormalisation(torch.autograd.Function):
    def forward(ctx, x, gamma, beta, epsilon):
        xn = x.detach().cpu().numpy()
        gamman = gamma.detach().cpu().numpy()
        betan = beta.detach().cpu().numpy()

        mean = xn.mean(axis=1, keepdims=True)
        var = xn.var(axis=1, keepdims=True)

        std = np.sqrt(var + epsilon)
        xhat = (xn - mean) / std

        y = gamman * xhat + betan

        ctx.save_for_backward(
            x,
            gamma,
            beta,
        )
        ctx.xhat = xhat
        ctx.std = std
        ctx.epsilon = epsilon

        return torch.from_numpy(y).to(x.device)

    def backward(ctx, dY):
        x, gamma, beta = ctx.saved_tensors
        xhat = ctx.xhat
        std = ctx.std
        epsilon = ctx.epsilon

        xn = x.detach().cpu().numpy()
        gamman = gamma.detach().cpu().numpy()
        betan = beta.detach().cpu().numpy()
        dYn = dY.detach().cpu().numpy()

        B, T, D = dYn.shape

        dbeta = np.sum(dYn, axis=(0, 1))

        dgamma = np.sum(dYn * xhat, axis=(0, 1))

        dxhat = dYn * gamman
        dstd = np.sum(dxhat * (xhat * -1) / (std), axis=-1, keepdims=True)
        dvar = np.sum(dxhat * (xhat * -0.5) / (std**2), axis=-1, keepdims=True)
        ddiff = dvar * 2 * xhat * std / (D) + dxhat / (std)
        dmean = np.sum(dxhat * -1 / (std), axis=-1, keepdims=True)
        dx = ddiff + dmean / D

        return (
            torch.from_numpy(dx).to(x.device),
            torch.from_numpy(dgamma).to(x.device),
            torch.from_numpy(dbeta).to(x.device),
            None,
        )
