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
model.eval()
with torch.no_grad():
    prompt = "The history of artificial intelligence"
    input_ids = tokeniser(prompt, return_tensors="pt")["input_ids"]
    generated = input_ids[:, :1]

    p = 0.9
    for _ in range(50):  # generate 50 tokens
        output = model(input_ids=generated)
        logits = output.logits
        probs = torch.softmax(logits[:, -1, :], dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Always keep at least the first token
        mask = cumulative_probs <= 0.9
        mask[:, 0] = True  # first token always included

        sorted_probs = sorted_probs * mask
        sorted_probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
        sorted_probs_sum = torch.clamp(
            sorted_probs_sum, min=1e-8
        )  # prevent division by zero
        sorted_probs = sorted_probs / sorted_probs_sum

        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)

        generated = torch.cat((generated, next_token), dim=1)


generated_text = tokeniser.decode(generated[0], skip_special_tokens=True)
print("Generated text:", generated_text)
