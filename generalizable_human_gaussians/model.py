from math import inf
import torch
from torch import nn
from torch.nn.modules import padding
import torchvision


class ResBlock(nn.Module):
    """docstring"""

    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        self.main_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        """docstring"""
        output = self.main_block(x) + self.skip(x)
        output = self.relu(output)
        return output


class Encoder(nn.Module):
    """docstring"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=5, stride=1, padding=1),
            ResBlock(in_channels=32, out_channels=32),
            ResBlock(in_channels=32, out_channels=32),
            ResBlock(in_channels=32, out_channels=48),
            ResBlock(in_channels=48, out_channels=48),
            ResBlock(in_channels=48, out_channels=96),
            ResBlock(in_channels=96, out_channels=96),
        )

    def forward(self, x: torch.Tensor):
        """docstring"""
        return self.mod(x)


class Decoder(nn.Module):
    """docstring"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            ResBlock(in_channels=96, out_channels=96),
            ResBlock(in_channels=96, out_channels=96),
            ResBlock(in_channels=96, out_channels=48),
            ResBlock(in_channels=48, out_channels=48),
            ResBlock(in_channels=48, out_channels=32),
            ResBlock(in_channels=32, out_channels=32),
        )

    def forward(self, x: torch.Tensor):
        """docstring"""
        return self.mod(x)
