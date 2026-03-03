import torch
import torch.nn as nn
import torch.optim as optim
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_model = 128
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_length = 256
dropout = 0.1

num_epochs = 10
lr = 3e-4

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token


loader, tokeniser = get_wikitext_dataloader(
    # Only training on 1000 values
    split="train",
    tokeniser_name="gpt2",
    batch_size=64,
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
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
optimiser = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(loader)
)

scaler = torch.amp.GradScaler("cuda")

model.train()


for epoch in range(num_epochs):
    total_loss = 0
    batches = 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        optimiser.zero_grad()
        with torch.amp.autocast("cuda"):
            output = model(src, tgt[:, :-1])

            loss = criterion(
                output.reshape(-1, vocab_size),
                tgt[:, 1:].reshape(-1),
            )

        if torch.isnan(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimiser)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        batches += 1

    avg_loss = total_loss / max(1, batches)

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    prompt = "Aircrafts are "
    input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids[:, :1]

    p = 0.9
    for _ in range(50):  # generate 50 tokens
        output = model(generated, generated[:, -1:])
        probs = torch.softmax(output[:, -1, :], dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs <= p
        mask[:, 0] = True

        sorted_probs = sorted_probs * mask
        sorted_probs_sum = torch.clamp(sorted_probs.sum(dim=-1, keepdim=True), min=1e-8)
        sorted_probs = sorted_probs / sorted_probs_sum

        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)

        generated = torch.cat((generated, next_token), dim=1)

generated_text = tokeniser.decode(generated[0], skip_special_tokens=True)
print("Generated text:", generated_text)
