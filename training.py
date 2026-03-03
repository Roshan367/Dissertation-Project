import torch
import torch.nn as nn
import torch.optim as optim
import math
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
d_model = 128
num_heads = 16
num_layers = 8
d_ff = 2048
max_seq_length = 128
dropout = 0.1
num_epochs = 10
lr = 3e-4

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

loader, tokeniser = get_wikitext_dataloader(
    split="train",
    tokeniser_name="gpt2",
    batch_size=64,
    max_length=max_seq_length,
)

vocab_size = tokeniser.vocab_size

# Decoder-only model — no src/tgt vocab split needed
model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout,
)
model.to(device)


def lr_lambda(step):
    warmup_steps = 4000
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


total_steps = num_epochs * len(loader)

criterion = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)

# Fixed betas for standard Adam — (0.9, 0.98) is for warmup schedulers only
optimiser = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#    optimiser, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(loader)
# )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

scaler = torch.amp.GradScaler("cuda")

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    batches = 0

    for batch in loader:
        # Decoder-only: single sequence, predict next token
        tgt = batch["tgt"].to(device)  # (B, T)
        inp = tgt[:, :-1]  # input:  tokens 0..T-2
        label = tgt[:, 1:]  # target: tokens 1..T-1

        optimiser.zero_grad()

        with torch.amp.autocast("cuda"):
            output = model(inp)  # (B, T-1, vocab_size)
            loss = criterion(
                output.reshape(-1, vocab_size),
                label.reshape(-1),
            )

        if not torch.isfinite(loss):
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
    perplexity = math.exp(avg_loss)  # Fixed: was torch.exp() on a float
    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")


# ── Inference ─────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    prompt = "Aircrafts are "
    input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids
    p = 0.9

    for _ in range(50):
        # Fixed: feed full growing context every step, not just last token
        output = model(generated)  # (1, T, vocab_size)
        probs = torch.softmax(output[:, -1, :], dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= p
        mask[:, 0] = True  # always keep top token
        sorted_probs = sorted_probs * mask
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )

        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)
        generated = torch.cat((generated, next_token), dim=1)

        # Stop at EOS
        if next_token.item() == tokeniser.eos_token_id:
            break

generated_text = tokeniser.decode(generated[0], skip_special_tokens=True)
print("Generated text:", generated_text)
