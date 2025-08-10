import math
import torch
from torch import div, nn
from torch._C import _pop_torch_dispatch_stack


class InputEmbeddings(nn.Module):
    """docstring"""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """docstring"""
        return self.embeddings(x) * math.sqrt(self.d_model)
