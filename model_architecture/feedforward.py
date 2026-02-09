import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NumpyFeedForward(torch.autograd.Function):
    def forward(ctx, x, W1, b1, W2, b2):
        xn = x.detach().cpu().numpy()
        W1n = W1.detach().cpu().numpy()
        b1n = b1.detach().cpu().numpy()
        W2n = W2.detach().cpu().numpy()
        b2n = b2.detach().cpu().numpy()

        z1 = xn @ W1n.T + b1n
        a1 = np.maximum(0, z1)

        y = z1 @ W2n.T + b2n

        ctx.save_for_backward(
            x,
            W1,
            b1,
            W2,
            b2,
        )
        ctx.z1 = z1

        return torch.from_numpy(y).to(x.device)

    def backward(ctx, dY):
        x, W1, b1, W2, b2 = ctx.saved_tensors
        z1 = ctx.z1

        xn = x.detach().cpu().numpy()
        W1n = W1.detach().cpu().numpy()
        W2n = W2.detach().cpu().numpy()
        dYn = dY.detach().cpu().numpy()

        da1 = dYn @ W2n
        dW2 = dYn.reshape(-1, dYn.shape[-1]).T @ np.maximum(0, z1).reshape(
            -1, z1.shape[-1]
        )
        db2 = dYn.sum(axis=(0, 1))

        dz1 = da1 * (z1 > 0)

        dx = dz1 @ W1n
        dW1 = dz1.reshape(-1, dz1.shape[-1]).T @ xn.reshape(-1, xn.shape[-1])
        db1 = dz1.sum(axis=(0, 1))

        return (
            torch.from_numpy(dx).to(x.device),
            torch.from_numpy(dW1).to(x.device),
            torch.from_numpy(db1).to(x.device),
            torch.from_numpy(dW2).to(x.device),
            torch.from_numpy(db2).to(x.device),
        )
