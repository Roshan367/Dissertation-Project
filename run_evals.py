# run_evals.py
import torch
import lm_eval
from transformers import AutoTokenizer
from model_architecture.transformer import Transformer
from eval_wrapper import TransformerEvalWrapper

# Hyperparameters
d_model = 512
num_heads = 8
num_layers = 8
d_ff = 2048
max_seq_length = 512
dropout = 0.1
num_epochs = 5
lr = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

vocab_size = tokeniser.vocab_size


model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=dropout,
)
model.load_state_dict(torch.load("models/transformer_best.pth", map_location=device))
model.to(device)

wrapped = TransformerEvalWrapper(model, tokeniser, device)

results = lm_eval.simple_evaluate(
    model=wrapped,
    tasks=["winogrande", "lambada_openai", "blimp", "hellaswag", "piqa"],
    num_fewshot=0,       # zero-shot; set to e.g. 5 for few-shot
    batch_size=1,
    device=str(device),
)

lm_eval.utils.make_table(results)


print("=== EVALUATION COMPLETE ===")
print(results["results"])

# Save to file as backup
import json
with open("eval_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)

print("Results saved to eval_results.json")