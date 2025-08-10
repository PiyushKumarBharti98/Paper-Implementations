import torch
from torch import nn
from . import multi_query_attention


class Decoder(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
