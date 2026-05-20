# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import time
from transformers import AutoTokenizer
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

model = Transformer(
    vocab_size=50257,
    d_model=512,
    num_heads=8,
    num_layers=8,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.0,
)
model.load_state_dict(torch.load("models/transformer_best.pth", map_location=device))
model.to(device)
model.eval()

# Test Perplexity
test_loader, _ = get_wikitext_dataloader(
    split="test",
    batch_size=8,
    max_length=512
)

total_loss = 0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=tokeniser.pad_token_id, reduction="sum")

with torch.no_grad():
    for batch in test_loader:
        ids = batch["tgt"].to(device)
        logits = model(ids)
        logits = logits[:, :-1, :].contiguous().view(-1, 50257)
        targets = ids[:, 1:].contiguous().view(-1)

        loss = criterion(logits, targets)

        non_pad = (targets != tokeniser.pad_token_id).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad

avg_loss = total_loss / max(1, total_tokens)
print(f"Avg loss: {avg_loss:.4f}")
perplexity = math.exp(avg_loss)
print(f"Test Perplexity: {perplexity:.2f}")

# Inference with multiple prompts
prompts = [
    "Aircrafts are ",
    "The Roman Empire fell because ",
    "Scientists have discovered that ",
    "The film was directed by ",
]

with torch.no_grad():
    for prompt in prompts:
        input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"].to(device)
        generated = input_ids
        p = 0.9
        temperature = 1.0
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        for _ in range(50):
            output = model(generated)          # ? plain tensor in, plain tensor out
            logits = output[:, -1, :]          # ? index directly, no .logits

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            shifted = torch.roll(cumulative_probs, 1, dims=-1)
            shifted[:, 0] = 0.0
            mask = shifted < p
            sorted_probs = sorted_probs * mask
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sampled_index = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_index)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokeniser.eos_token_id:
                break
                

        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        
        elapsed = end - start
        
        generated_text = tokeniser.decode(generated[0], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_text}")
        print(f"Inference time: {elapsed:.3f}s")
