import torch
from torch import nn
from torch.nn.modules import padding
from ast import increment_lineno
import torch
from torch import nn
from torch.nn.modules import padding


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


class GatedConv(nn.Module):
    """docstring"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.mask = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x):
        """docstring"""
        features = self.conv(x)
        mask = torch.sigmoid(self.mask(x))
        return features * mask


class Encoder(nn.Module):
    """docstring"""

    def __init__(self, in_channels) -> None:
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

    def __init__(self) -> None:
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


class MapsGenerator(nn.Module):
    """docstring"""

    def __init__(self, geometric_channels: int, appearance_channels: int) -> None:
        super().__init__()
        self.geometry_encoder = Encoder(in_channels=geometric_channels)
        self.appearance_encoder = Encoder(in_channels=appearance_channels)
        self.decoder = Decoder()

        self.rotation_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1),
        )

        self.scaling_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
        )

        self.opacity_head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, geometric_channels, appearance_channels):
        """docstring"""
        enc_geo = self.geometry_encoder(geometric_channels)
        enc_appr = self.appearance_encoder(appearance_channels)

        combined_features = enc_geo + enc_appr
        decoded_features = self.decoder(combined_features)

        rotation_map = self.rotation_head(decoded_features)
        scaling_map = self.scaling_head(decoded_features)
        opacity_map = self.opacity_head(decoded_features)

        return rotation_map, scaling_map, opacity_map


class SelfAttention(nn.Module):
    """docstring"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """docstring"""
        batch_size, C, width, height = x.size()

        proj_query = self.query(x).view(batch_size, -1, widht * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)

        attention = torch.bmm(proj_query, proj_key)

        attention = torch.nn.functional.softmax(attention, dim=1)

        proj_value = self.value(x).view(batch_size, -1, width * height).permute(0, 2, 1)

        output = torch.bmm(attention, proj_value).view(batch_size, C, width, height)

        return output


class InPaintingGenerator(nn.Module):
    """docstring"""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        self.coarse_encoder = nn.Sequential(
            GatedConv(in_channels, 48, kernel_size=5, stride=1, padding=2),
            GatedConv(48, 48, kernel_size=3, stride=1, padding=1),
            GatedConv(48, 96, kernel_size=3, stride=2, padding=1),
            GatedConv(96, 96, kernel_size=3, stride=1, padding=1),
            GatedConv(96, 192, kernel_size=3, stride=2, padding=1),
            GatedConv(192, 192, kernel_size=3, stride=1, padding=1),
        )

        self.coarse_dilation = nn.Sequential(
            GatedConv(192, 192, kernel_size=3, stride=1, padding=2, dilation=2),
            GatedConv(192, 192, kernel_size=3, stride=1, padding=4, dilation=4),
            GatedConv(192, 192, kernel_size=3, stride=1, padding=8, dilation=8),
            GatedConv(192, 192, kernel_size=3, stride=1, padding=16, dilation=16),
        )

        self.coarse_decoder = nn.Sequential(
            GatedConv(192, 192, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedConv(192, 96, kernel_size=3, stride=1, padding=1),
            GatedConv(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedConv(96, 48, kernel_size=3, stride=1, padding=1),
            GatedConv(48, 24, kernel_size=3, stride=1, padding=1),
            GatedConv(24, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(3),
        )
