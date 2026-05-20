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

    """
    Forward paass for the decoder which acts as the residual layer
    """
    def forward(self, x, mask):
        # Causal self-attention (Pre-LN)
        normed_x = self.norm1(x)
        attention_output = self.self_attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attention_output)
        # Feedforward (Pre-LN)
        ff_output = self.feedforward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x
