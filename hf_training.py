import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer, AutoModelForCausalLM

max_seq_length = 64
num_epochs = 3

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokeniser))

loader, tokeniser = get_wikitext_dataloader(
    # Only training on 1000 values
    split="train[:1000]",
    tokeniser_name="gpt2",
    batch_size=2,
    max_length=max_seq_length,
)

optimiser = optim.AdamW(model.parameters(), lr=0.0001)

model.train()


for epoch in range(num_epochs):
    for i, batch in enumerate(loader):
        if i > 5:
            break

        src = batch["src"]

        outputs = model(input_ids=src, labels=src)

        loss = outputs.loss
        optimiser.zero_grad()
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
