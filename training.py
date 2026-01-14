import torch
import torch.nn as nn
import torch.optim as optim
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer

# Tiny settings for fast test
d_model = 128
num_heads = 4
num_layers = 2
d_ff = 512
max_seq_length = 64
dropout = 0.1
num_epochs = 3

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token


loader, tokeniser = get_wikitext_dataloader(
    split="train[:1000]",
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
optimiser = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()


for epoch in range(num_epochs):
    for batch in loader:
        src = batch["src"]
        tgt = batch["tgt"]

        optimiser.zero_grad()
        output = model(src, tgt[:, :-1])

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1),
        )
        loss.backward()
        optimiser.step()
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    prompt = "The history of artificial intelligence"
    input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids[:, :1]

    p = 0.9
    for _ in range(50):  # generate 50 tokens
        output = model(generated, generated)
        probs = torch.softmax(output[:, -1, :], dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative_probs > p] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)

        generated = torch.cat((generated, next_token), dim=1)

generated_text = tokeniser.decode(generated[0], skip_special_tokens=True)
print("Generated text:", generated_text)
