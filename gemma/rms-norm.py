import torch
from torch import nn


class RMSNorm(nn.Module):
    """docstring"""

    def __init__(self, dim, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """docstring"""
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
