import torch
import torch.nn as nn
import torch.optim as optim
import math
from model_architecture.transformer import Transformer
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

model = Transformer()
model.load_save_dict(torch.load("models/transformer_model.pth"))


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
