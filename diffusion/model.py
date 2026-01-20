from _typeshed import Self
import math
import torch
from torch import channels_last, conv2d, diagonal, nn, scalar_tensor
import torch.nn.functional as F
from einops import rearrange, einsum
from torch.nn.modules import padding


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        input_projection_bias: bool,
        output_projection_bias: bool,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=input_projection_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=output_projection_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, masked=False):

        batch_size, seq_len, d_embed = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # einops operations
        # q = rearrange(q, "b s (h  d) -> b h s d", h=self.n_heads)
        # k = rearrange(k, "b s (h  d) -> b h s d", h=self.n_heads)
        # v = rearrange(v, "b s (h  d) -> b h s d", h=self.n_heads)
        #
        # qk = einsum(q, k, "b h i d, b h j d -> b h i j") / math.sqrt(self.d_head)
        #
        # if masked == True:
        #     mask = torch.ones(seq_len, seq_len, dtype=bool, device=qk.device).triu(
        #         diagonal=1
        #     )
        #     qk = qk.masked_fill(mask)
        #
        # mul = F.softmax(qk)
        # final = einsum(qk, v, "b h i j, b h j d -> b h i d")
        #
        # output = rearrange(final, "b h s d -> b s (h d)")
        #
        # output = self.out_proj(output)

        # manual forward pass
        q = q.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        k = k.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        v = v.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        scale = math.sqrt(self.d_head)

        # qk = torch.matmul(q, k).view(
        #     self.batch_size, self.n_heads, self.seq_len, self.seq_len
        # )

        qk = torch.matmul(q, k).transpose(-1, -2)

        if masked == True:
            mask = torch.ones(seq_len, seq_len, dtype=bool, device=qk.device).triu(
                diagonal=1
            )
            qk = qk.masked_fill(mask)

        mul = qk / scale

        final = F.softmax(qk)

        output = torch.matmul(final, v)

        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        input_projection_bias: bool,
        output_projection_bias: bool,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=input_projection_bias)
        self.k_proj = nn.Linear(d_embed, d_embed, bias=input_projection_bias)
        self.v_proj = nn.Linear(d_embed, d_embed, bias=input_projection_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=output_projection_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y, masked=False):

        batch_size, seq_len, d_embed = x.shape

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # einops operations
        # q = rearrange(q, "b s (h  d) -> b h s d", h=self.n_heads)
        # k = rearrange(k, "b s (h  d) -> b h s d", h=self.n_heads)
        # v = rearrange(v, "b s (h  d) -> b h s d", h=self.n_heads)
        #
        # qk = einsum(q, k, "b h i d, b h j d -> b h i j") / math.sqrt(self.d_head)
        #
        # if masked == True:
        #     mask = torch.ones(seq_len, seq_len, dtype=bool, device=qk.device).triu(
        #         diagonal=1
        #     )
        #     qk = qk.masked_fill(mask)
        #
        # mul = F.softmax(qk)
        # final = einsum(qk, v, "b h i j, b h j d -> b h i d")
        #
        # output = rearrange(final, "b h s d -> b s (h d)")
        #
        # output = self.out_proj(output)

        # manual forward pass
        q = q.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        k = k.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        v = v.view(self.batch_size, self.seq_len, self.n_heads, self.d_head).transpose(
            1, 2
        )

        scale = math.sqrt(self.d_head)

        # qk = torch.matmul(q, k).view(
        #     self.batch_size, self.n_heads, self.seq_len, self.seq_len
        # )

        qk = torch.matmul(q, k).transpose(-1, -2)

        if masked == True:
            mask = torch.ones(seq_len, seq_len, dtype=bool, device=qk.device).triu(
                diagonal=1
            )
            qk = qk.masked_fill(mask)

        mul = qk / scale

        final = F.softmax(qk)

        output = torch.matmul(final, v)

        output = self.out_proj(output)

        return output


class ClIPEmbeddings(nn.Module):
    def __init__(self, n_heads: int, d_embed: int) -> None:
        super().__init__()
        self.attention = SelfAttention(n_heads, d_embed, True, True)
        self.postional_embeddings = nn.Parameter(torch.zeros(n_heads, d_embed))


class VAE_AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_projection_bias: bool = True,
        output_projection_bias: bool = True,
    ) -> None:
        super().__init__()
        self.GroupNorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(
            1, in_channels, input_projection_bias, output_projection_bias
        )

    def forward(self, x):

        residue = x

        batch_size, channels, height, width = x.size

        x = x.view(batch_size, channels, height * width).transpose(1, 2)

        x = self.attention(x)

        x = x.transpose(1, 2)

        x = x.view(batch_size, channels, height, width)

        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.GroupNorm1 = nn.GroupNorm(32, in_channels)
        self.Convlutional1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.GroupNorm2 = nn.GroupNorm(32, out_channels)
        self.Convlutional2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=1,
            )

    def forward(self, x):

        residual = x

        x = self.GroupNorm1(x)
        x = self.Convlutional1(x)
        x = self.GroupNorm2(x)
        x = self.Convlutional2(x)

        return x + self.residual_layer(residual)


class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
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
            ]
        )

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


class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(4, 512, kernel_size=1, padding=0),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                VAE_ResidualBlock(256, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                nn.GroupNorm(32, 128),
                nn.SiLU(),
                nn.Conv2d(128, 3, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x, noise):
        x /= 0.18215
        x = self.layers(x)
        return x
