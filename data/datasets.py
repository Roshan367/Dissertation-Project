from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

"""
Class used to load datasets properly depending on the dataset, split,
and tokeniser
"""


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokeniser, max_length=128):
        self.texts = texts
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokeniser(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0)

        return {
            "src": ids,
            "tgt": ids.clone(),
        }


def get_wikitext_dataloader(
    split="train", tokeniser_name="gpt2", batch_size=8, max_length=64
):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = dataset["text"]

    tokeniser = AutoTokenizer.from_pretrained(tokeniser_name)
    tokeniser.pad_token = tokeniser.eos_token

    dataset = WikiTextDataset(texts, tokeniser, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, tokeniser
