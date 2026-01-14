import torch
import torch.nn as nn
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader

# Tiny settings for fast test
d_model = 128
num_heads = 4
num_layers = 2
d_ff = 512
max_seq_length = 64
dropout = 0.1

loader, tokeniser = get_wikitext_dataloader(
    split="train[:10]",
    tokeniser_name="gpt2",
    batch_size=2,
    max_length=max_seq_length,
)

vocab_size = tokeniser.vocab_size

model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout,
)

criterion = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)

model.train()

# ---- SINGLE BATCH TEST ----
batch = next(iter(loader))

src = batch["src"]
tgt = batch["tgt"]

output = model(src, tgt[:, :-1])

loss = criterion(
    output.reshape(-1, vocab_size),
    tgt[:, 1:].reshape(-1),
)

print("Test loss:", loss.item())
print("Output shape:", output.shape)
