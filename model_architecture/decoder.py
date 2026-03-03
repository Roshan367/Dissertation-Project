import torch
import torch.nn as nn
from . import normalisation
from . import attention
from . import feedforward

"""
Decoder block for decoder-only Transformer.
Cross-attention removed — only causal self-attention + feedforward.
"""


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.self_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.feedforward = feedforward.FeedForward(d_model, d_ff)
        self.norm1 = normalisation.HybridLayerNorm(d_model)
        self.norm2 = normalisation.HybridLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Causal self-attention
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
