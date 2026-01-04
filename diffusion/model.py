import torch
from torch import nn
import einops


class Attention(nn.Module):
    def __init__(self, n_heads: int, seq_len: float) -> None:
        super().__init__()


class Encoder(nn.Module):
    def __init__(
        self,
        image: torch.Tensor,
        noise: float,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(Attention)
