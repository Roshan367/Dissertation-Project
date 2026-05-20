import torch
import torch.nn as nn
import torch.optim as optim
import math
from accelerate import Accelerator
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel

accelerator = Accelerator()

d_model = 512
num_heads = 8
num_layers = 8
d_ff = 2048
max_seq_length = 512
dropout = 0.1
num_epochs = 5
lr = 3e-4

with accelerator.main_process_first():
    tokeniser = AutoTokenizer.from_pretrained("gpt2")
    tokeniser.pad_token = tokeniser.eos_token
    loader, tokeniser = get_wikitext_dataloader(
        split="train",
        tokeniser_name="gpt2",
        batch_size=64,
        max_length=max_seq_length,
    )
    val_loader, _ = get_wikitext_dataloader(
        split="validation",
        tokeniser_name="gpt2",
        batch_size=64,
        max_length=max_seq_length,
    )

vocab_size = tokeniser.vocab_size

config = GPT2Config(
    n_embd=512,
    n_head=8,
    n_layer=8,
    n_positions=512,
    n_ctx=512,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    n_inner=2048,
    vocab_size=vocab_size
)
model = GPT2LMHeadModel(config)

if accelerator.is_main_process:
    print(f"Layer norm weights (should be 1.0 if random init): {model.transformer.h[0].ln_1.weight[:5]}")

def lr_lambda(step):
    warmup_steps = 4000
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

total_steps = num_epochs * len(loader)
criterion = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id)
optimiser = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

model, optimiser, loader, scheduler = accelerator.prepare(
    model, optimiser, loader, scheduler
)
val_loader = accelerator.prepare(val_loader)

best_val_loss = float("inf")
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    batches = 0
    for batch in loader:
        tgt = batch["tgt"]
        inp = tgt[:, :-1]
        label = tgt[:, 1:]

        optimiser.zero_grad()
        with accelerator.autocast():
            output = model(input_ids=inp)
            loss = criterion(output.logits.reshape(-1, vocab_size), label.reshape(-1))

        if not torch.isfinite(loss):
            continue

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()
        total_loss += loss.item()
        batches += 1

    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            tgt = batch["tgt"]
            inp = tgt[:, :-1]
            label = tgt[:, 1:]
            with accelerator.autocast():
                output = model(input_ids=inp)
                loss = criterion(output.logits.reshape(-1, vocab_size), label.reshape(-1))
            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / max(1, val_batches)
    val_perplexity = math.exp(avg_val_loss)
    model.train()
    avg_loss = total_loss / max(1, batches)
    perplexity = math.exp(avg_loss)

    if accelerator.is_main_process:
        print(
            f"Epoch {epoch + 1} - Train PPL: {perplexity:.2f} | Val PPL: {val_perplexity:.2f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            accelerator.unwrap_model(model).save_pretrained("models/hf_gpt2_best")
            print(f"New best model saved (Val Loss: {avg_val_loss:.4f})")