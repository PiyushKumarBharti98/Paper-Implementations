import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """docstring"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.layer2(self.dropout(nn.GELU(self.layer1(x))))
