import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = CustomScaledDotAttention.apply(Q, K, V, mask, self.d_k)
        output = self.W_o(self.combine_heads(attention_output))
        return output


class CustomScaledDotAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, mask, d_k):
        scale = 1.0 / math.sqrt(d_k)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_scores = (
            attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        )
        exp_scores = torch.exp(attention_scores)
        P = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

        O = torch.matmul(P, V)

        ctx.save_for_backward(
            Q,
            K,
            V,
            P,
            mask,
        )
        ctx.d_k = d_k

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P, mask = ctx.saved_tensors
        d_k = ctx.d_k

        scale = 1.0 / math.sqrt(d_k)

        dV = torch.matmul(P.transpose(-2, -1), dO)

        dP = torch.matmul(dO, V.transpose(-2, -1))

        dS = dP - (dP * P).sum(dim=-1, keepdim=True)
        dS *= P
        dS *= scale

        if mask is not None:
            dS = dS.masked_fill(mask == 0, 0.0)

        dQ = torch.matmul(dS, K)

        dK = torch.matmul(dS.transpose(-2, -1), Q)

        return (
            dQ,
            dK,
            dV,
            None,
            None,
        )
