from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class HFDataset(Dataset):
    def __init__(self, hf_dataset, tokeniser, max_length=128):
        self.dataset = hf_dataset
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        source = item["source"]
        target = item["target"]

        src_enc = self.tokeniser(
            source,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tgt_enc = self.tokeniser(
            target,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "src": src_enc["input_ids"].squeeze(0),
            "tgt": tgt_enc["input_ids"].squeeze(0),
        }


def get_wikitext_dataloader(
    split="train", tokeniser_name="gpt2", batch_size=32, max_length=128
):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = dataset["text"]

    tokeniser = AutoTokenizer.from_pretrained("tokeniser_name")
    tokeniser.pad_token = tokeniser.eos_token

    dataset = HFDataset(texts, tokeniser, max_length=max_length)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, tokeniser
