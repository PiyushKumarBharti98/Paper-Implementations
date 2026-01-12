import torch
from torch import conv2d, nn
from torch._higher_order_ops.associative_scan import assoiciative_scan_fake_tensor_mode
from torch.nn.modules import Conv2d, padding
import einops


class Attention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.GroupNorm1 = nn.GroupNorm(32, in_channels)
        self.Convlutional1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.GroupNorm2 = nn.GroupNorm(32,out_channels)
        self.Convlutional2 = nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1
                )

        if in_channels==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,padding=1)

    def forward(self,x):

        residual = x

        x = self.GroupNorm1(x)
        x = self.Convlutional1(x)
        x = self.GroupNorm2(x)
        x = self.Convlutional2(x)

        return x + self.residual_layer(residual)


class Encoder(nn.Module):
    def __init__(
        self,
        image: torch.Tensor,
        noise: float,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), 
            ResidualBlock(128, 128),
            ResidualBlock(128,128),
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            ResidualBlock(128,256),
            ResidualBlock(256,256),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            ResidualBlock(256,512),
            ResidualBlock(512,512),
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            ResidualBlock(512,512),
            ResidualBlock(512,512),
            ResidualBlock(512,512),
            Attention(512),
            nn.GroupNorm(32,512),
            nn.SiLU()
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            nn.Conv2d(8,8,kernel_size=3,padding=0)
        )
