import torch
import torch.nn as nn
import math
from . import decoder
from . import encoding

"""
Transformer class - Decoder-only architecture for language modelling.
Takes a single sequence and predicts the next token at each position.
"""


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = encoding.PositionalEncoding(d_model, max_seq_length)
        self.decode_layers = nn.ModuleList(
            [
                decoder.Decoder(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, x):
        device = x.device
        seq_length = x.size(1)
        # Padding mask: (B, 1, 1, T)
        pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        # Causal mask: (1, T, T) — prevents attending to future tokens
        causal_mask = (
            1
            - torch.triu(
                torch.ones(1, seq_length, seq_length, device=device), diagonal=1
            )
        ).bool()
        # Combine: (B, 1, T, T)
        return pad_mask & causal_mask

    def forward(self, x):
        mask = self.generate_mask(x)
        x = self.dropout(self.positional_encoding(self.embedding(x)))
        for layer in self.decode_layers:
            x = layer(x, mask)
        return self.fc(x)
