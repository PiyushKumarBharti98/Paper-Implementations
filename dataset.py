import torch
from torch import nn
from torch.utils.data import Dataset


class LoadDataset(Dataset):

    def __init__(self, ds, input_token, output_token, input, output, seq_len):
        super().__init__()
        self.ds = ds
        self.input_token = input_token
        self.output_token = output_token
        self.input = input
        self.output = output
        self.seq_len = seq_len

        self.sos_token = torch.Tensor(
            [input_token.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.Tensor(
            [input_token.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.Tensor(
            [input_token.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        input_target = self.ds[idx]
        input_text = input_target["translation"][self.input]
        output_text = input_target["translation"][self.output]

        encoder_input_tokens = self.input_token.encode(input_text).ids
        decoder_input_tokens = self.input_token.decode(output_text).ids

        encoder_padding = self.seq_len - len(encoder_input) - 2
        decoder_padding = self.seq_len - len(decoder_input) - 1

        if encoder_padding < 0 or decoder_padding < 0:
            raise ValueError("sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * encoder_padding, dtype=torch.int64),
            ],
            dim=0,
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(decoder_input_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token] * decoder_padding, dtype=torch.int64),
            ],
            dim=0,
        )
       label = torch.cat(
            [
                torch.Tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * decoder_padding, dtype=torch.int64),
            ],
            dim=0,
        )

    assert encoder_input.size(0) == self.seq_len
    assert decoder_padding.size(0)== self.seq_len
    assert label.size(0) == self.seq_len
    
    return{
        "encoder_input":encoder_input,
        "decoder_input":decoder_input,
        "encoder_mask":,
        "decoder_mask":,
        "label":label,
        "input_text":input_text,
        "output_text":output_text,
    }
