# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import time
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from data.datasets import get_wikitext_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

# Load HF GPT-2 model
model = GPT2LMHeadModel.from_pretrained("models/hf_gpt2_best")
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

        # ? HF model takes input shifted — pass full sequence, use labels for loss
        outputs = model(input_ids=ids[:, :-1])
        logits = outputs.logits.contiguous().view(-1, tokeniser.vocab_size)
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
    "The football match ended ",
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
            # ? HF model returns an object, extract logits
            output = model(input_ids=generated)
            logits = output.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            probs = torch.softmax(logits, dim=-1)

            # Top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Right shift so token that first exceeds p is still included
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