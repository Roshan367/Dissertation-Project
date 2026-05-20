import torch
import lm_eval
import json
import math
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from eval_wrapper import TransformerEvalWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("models/hf_gpt2_best")
model.to(device)
model.eval()

class HFTransformerEvalWrapper(TransformerEvalWrapper):
    def _model_call(self, inps):
        with torch.no_grad():
            output = self.model(input_ids=inps)
            return output.logits  

wrapped = HFTransformerEvalWrapper(model, tokenizer, device)

results = lm_eval.simple_evaluate(
    model=wrapped,
    tasks=["blimp", "hellaswag", "winogrande", "piqa"],
    num_fewshot=0,
    batch_size=1,
    device=str(device),
)

print("=== HF GPT2 EVALUATION COMPLETE ===")
print(results["results"])

with open("hf_eval_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)

print("Results saved to hf_eval_results.json")