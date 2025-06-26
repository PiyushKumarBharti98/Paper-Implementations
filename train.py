from torch.nn import nn
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from . import dataset
from dataset import LoadDataset
from pathlib import Path
import torch
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



def greedy_decode(something):
    return something

def get_all_sentences(data, language):
    """docstring"""
    for item in data:
        return item["translation"][language]

def run_validation(model, val_ds, input_tokenzier, output_tokenzier, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count=0

    source = []
    expected_output = []
    predictions = []

    with torch.no_grad():
        for batch in val_ds:
            count += 1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0)==1

            model_output = greedy_decode()

            input_text = batch["lang_src"][0]
            output_text = batch["lang_tgt"][0]
            model_output_text = output_tokenzier.decode(model_output.detach().cpu().numpy())

            source.append(input_text)
            expected_output.append(output_text)
            predictions.append(model_output_text)

            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

            if writer:
                metric = torchmetrics.CharErrorRate()
                cer = metric(predicted, expected_output)
                writer.add_scalar('validation cer', cer, global_step)
                writer.flush()

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(predicted, expected_output)
                writer.add_scalar('validation wer', wer, global_step)
                writer.flush()

                # Compute the BLEU metric
                metric = torchmetrics.BLEUScore()
                bleu = metric(predicted, expected_output)
                writer.add_scalar('validation BLEU', bleu, global_step)
                writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

    

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
    raw_data = load_dataset("Helsinki-NLP/opus_books", "en-pl")

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


def get_model(config, input_vocab_len, output_vocab_len):
    """docstring"""
    model = build_transformer(
        input_vocab_len,
        output_vocab_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def train_model(config):
    device = "cuda"
    device = torch.device(device)

    train_dataloader, val_dataloader, input_tokenzier, output_tokenzier = get_dataset(
        config
    )
    model = get_model(
        config, input_tokenzier.get_vocab_size(), output_tokenzier.get_vocab_size()
    ).to(device)

    writer = SummaryWriter(config["experiment"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    start_epoch = 0
    g_step = 0
    preload = config["preload"]
    model_file = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload) if preload else None
    )

    if model_file:
        print(f"preloading model {model_file}")
        state = torch.load(model_file)
        start_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        g_step = state["g_step"]
    else:
        print("no model preload")

    loss_func = nn.CrossEntropyLoss(
        ignore_index=input_tokenzier.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(start_epoch, config["num_epoch"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iteration = tqdm(train_dataloader, desc=f"processing epoch {epoch:02d}")
        for batch in batch_iteration:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_input, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)
            loss = loss_func(
                proj_output.view(-1, output_tokenzier.get_vocab_size()), label.view(-1)
            )
            batch_iteration.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss ", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model,val_dataloader,input_tokenzier,output_tokenzier,config['seq_len'],device,lambda msg:batch_iteration.write(msg),global_step,writer)

        model_filename = get_model_file_path(config,f"{epoch:02d}")
        torch.save({
            'epoch':epoch,
            'model_state_dictionary':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        },model_filename)

if __name__ = "__main__":
    warnings.filterwarnings("ignore")
    config.get_config()
    train_model(config)
