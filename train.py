from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from . import dataset
from dataset import LoadDataset
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR  # Still widely used

import warnings
from tqdm import tqdm
import os

from datasets import (
    load_dataset,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import technometrics
from torch.utils.tensorboard import (
    SummaryWriter,
)


def get_all_sentences(data, language):
    """docstring"""
    for item in data:
        return item["translation"][language]


def build_tokenizer(config, data, language):
    """docstring"""
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(data, language), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    """docstring"""
    raw_data = load_dataset("Helsinki-NLP/opus_books", "ca-de")

    input_tokenzier = build_tokenizer(config, raw_data, config["lang_src"])
    output_tokenzier = build_tokenizer(config, raw_data, config["lang_tgt"])

    train_size = int(len(raw_data) * 0.9)
    val_size = len(raw_data) - train_size
    train_data, val_data = random_split(raw_data, [train_size, val_size])

    train_ds = LoadDataset(
        train_data,
        input_tokenzier,
        output_tokenzier,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = LoadDataset(
        val_data,
        input_tokenzier,
        output_tokenzier,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    input_max_len = 0
    output_max_len = 0
    for item in raw_data:
        input_ids = input_tokenzier.encode(item["translation"][config["lang_src"]]).ids
        output_ids = output_tokenzier.encode(
            item["translation"][config["lang_tgt"]]
        ).ids
        input_max_len = max(input_max_len, len(input_ids))
        output_max_len = max(output_max_len, len(output_ids))

    print(f"max len of input sentence is {input_max_len}")
    print(f"max len of output sentence is {output_max_len}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config("batch_size"), shuffle=True
    )

    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, input_tokenzier, output_tokenzier
