import torch
from torch import conv2d, nn
from torch._higher_order_ops.associative_scan import assoiciative_scan_fake_tensor_mode
from torch.nn.modules import Conv2d, padding
import einops


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()


class VAE_ResidualBlock(nn.Module):
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


class VAE_Encoder_ModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        ])
        
        self.needs_padding = [3, 6, 9]  
    
    def forward(self, x, noise):
        for i, layer in enumerate(self.layers):
            if i in self.needs_padding:
                _, _, h, w = x.shape
                pad_h = (2 - h % 2) % 2
                pad_w = (2 - w % 2) % 2
                x = F.pad(x, (0, pad_w, 0, pad_h))
            
            x = layer(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        x = mean + stdev * noise
        x *= 0.18215
        
        return x
