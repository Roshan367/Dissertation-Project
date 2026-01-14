import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets import get_wikitext_dataloader
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokeniser = AutoTokenizer.from_pretrained(model_name)
tokeniser.pad_token = tokeniser.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokeniser))
