from multiprocessing.spawn import prepare
from fsspec.implementations.local import stat
from huggingface_hub import DiscussionEvent, TokenClassificationInput
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math

from transformers.models.gemma3.modeling_gemma3 import token_type_ids_mask_function
from model import GemmaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "vocab_size": 30522,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "seq_len": 256,
}

TRAIN_CONFIG = {
    "batch_size": 16,
    "num_epochs": 3,
    "lr": 1e-4,
    "gradient_accumulation_steps": 4,
    "log_steps": 10,
}

print(f"Using device: {DEVICE}")


class TextDataset(Dataset):
    def __init__(self, token_id, seq_len) -> None:
        self.token_id = token_id
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_id) // self.seq_len

    def __getitem__(self, index):
        start_idx = index * self.seq_len
        end_idx = start_idx + self.seq_len

        input_ids = torch.tensor(self.token_id[start_idx:end_idx], dtype=torch.long)
        target_ids = torch.tensor(
            self.token_id[start_idx + 1 : end_idx + 1], dtype=torch.long
        )
        return input_ids, target_ids


def data_preparation(seq_len):
    """docstring"""
    print("preparing data .....")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    print("loading dataset from load dataset .....")
    dataset = load_dataset("mindchain/wikitext2", split="train")

    print("concating text")
    text = "\n".join(filter(None, dataset["text"]))
    token_ids = tokenizer.encode(text)
    print(f"len of token or no of tokens {len(token_ids)}")

    train_dataset = TextDataset(token_ids, seq_len)
    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True
    )

    print("dataset preparation complete")
    return train_loader, tokenizer.vocab_size


def training_loop():
    """docstring"""
    train_loader, vocab_size = data_preparation(MODEL_CONFIG["vocab_size"])
    MODEL_CONFIG["vocab_size"] = vocab_size

    model = GemmaModel(**MODEL_CONFIG).to(DEVICE)

    print("model architecture...\n")
    print(model)

    optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print("\n ---starting the training---\n")
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")

        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            mask = (
                torch.tril(torch.ones(MODEL_CONFIG["seq_len"], MODEL_CONFIG["seq_len"]))
                .view(1, 1, MODEL_CONFIG["seq_len"], MODEL_CONFIG["seq_len"])
                .to(DEVICE)
            )

            with torch.autocast(
                device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")
            ):
                logits = model(inputs, mask=mask)
                loss = loss_fn(
                    logits.view(-1, MODEL_CONFIG["vocab_size"]), targets.view(-1)
                )
                loss = loss / TRAIN_CONFIG["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            total_loss += loss.item()

            if (step + 1) % TRAIN_CONFIG["gradient_accumulation_steps"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % TRAIN_CONFIG["log_steps"]:
                avg_loss = total_loss * TRAIN_CONFIG["gradient_accumulation_steps"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                total_loss = 0

    print("training complete....")


if __name__ == "__main__":
    training_loop()
