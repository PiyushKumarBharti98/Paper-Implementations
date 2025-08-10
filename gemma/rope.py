import math
import torch
from torch import nn


class RoPositionalEncoding(nn.Module):
    """docstring"""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0)  # Add batch dimension: (1, seq_len, d_model)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        """docstring"""
        pos_emb = self.pe[:, : x.size(1)].requires_grad_(False)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        sin = pos_emb[..., 0::2, :]
        cos = pos_emb[..., 1::2, :]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x2 * sin

        x = torch.zeros_like(x)
        x[..., 0::2] = rotated_x1
        x[..., 1::2] = rotated_x2

        return self.dropout(x)
