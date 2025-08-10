import math
from torch import nn
import torch


class InputEmbedding(nn.Module):
    """docstring"""

    def __init__(self, vocab_size: int, d_model: int):
        """docstring"""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        return self.embedding(x) * math.sqrt(self.d_model)
