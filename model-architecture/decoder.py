import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import attention
import encoding
import feedforward
import decoder


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.self_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.cross_attention = attention.MultiHeadAttention(d_model, num_heads)
        self.feedforward = feedforward.FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.dropout(dropout)

    def forward(self, x, emb_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.drouput(attention_output))
        attention_output = self.cross_attention(x, emb_output, emb_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
