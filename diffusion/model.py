from _typeshed import Self
import math
import torch
from torch import channels_last, conv2d, diagonal, isin, nn, scalar_tensor, t
from torch._higher_order_ops.flex_attention import create_fw_bw_graph
from torch.autograd import forward_ad
import torch.nn.functional as F
from einops import rearrange, einsum
from torch.nn.modules import padding


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        input_projection_bias: bool = True,
        output_projection_bias: bool = True,
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
        d_cross: int,
        input_projection_bias: bool = True,
        output_projection_bias: bool = True,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=input_projection_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=input_projection_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=input_projection_bias)
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
    def __init__(self, n_vocab: int, d_embed: int, n_token: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(n_vocab, d_embed)
        self.postional_embeddings = nn.Parameter(torch.zeros(n_vocab, d_embed))

    def forward(self, x):

        x = self.token_embeddings(x)
        x += self.postional_embeddings

        return x


class CLIPPlayer(nn.Module):
    def __init__(self, d_embed: int, n_head: int) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(d_embed, n_head)
        self.layernorm2 = nn.LayerNorm(d_embed)

        self.linear1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x):

        residue = x

        x = self.layernorm1(x)
        x = self.attention(x)
        x += residue

        residue = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self, n_tokens: int, n_embed: int, n_head: int) -> None:
        super().__init__()
        self.embeddings = ClIPEmbeddings(n_tokens, n_embed, n_head)
        self.clipmodel = nn.ModuleList([CLIPPlayer(n_embed, n_head) for i in range(12)])
        self.layernorm = nn.LayerNorm(n_embed)

    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        x = self.embeddings(tokens)

        for layer in self.clipmodel:
            x = layer(x)

        out = self.layernorm(x)

        return out


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


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x):
        x = self.linear1(x)

        x = nn.SiLU(x)

        x = self.linear2(x)

        return x


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embed: int, d_cross: int = 768) -> None:
        super().__init__()
        channels = n_heads * n_embed

        self.groupnorm = nn.GroupNorm(32, channels)
        self.convlayer = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.selfattention = SelfAttention(n_heads, n_embed)

        self.layernorm2 = nn.LayerNorm(channels)
        self.crosssattention = CrossAttention(n_heads, n_embed)

        self.layernorm3 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, 4 * channels)
        self.linear2 = nn.Linear(4 * channels, channels)

        self.convout = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):

        long_residue = x

        x = self.groupnorm(x)
        x = self.convlayer(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h * w).transpose(-1, -2)

        short_residue = x

        x = self.layernorm1(x)
        x = self.selfattention(x)

        x += short_residue

        short_residue = x

        x = self.layernorm2(x)
        x = self.crosssattention(x, context)

        x += short_residue

        short_residue = x

        x = self.layernorm3(x)
        x, gate = self.linear1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear2(x)

        x += short_residue

        x = x.transpose(-1, -2).view((n, c, h, w))

        x = self.convout(x)

        return x + long_residue


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.convlayer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.timelayer = nn.Linear(n_time, out_channels)

        self.groupnorm_time = nn.GroupNorm(32, in_channels)
        self.convlayer_time = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, image: torch.Tensor, time):

        residue = image

        image = self.groupnorm(image)
        image = F.silu(image)
        image = self.convlayer(image)

        time = self.timelayer(time)

        y = image + time.unsqueeze(-1).unsqueeze(-1)
        y = self.groupnorm_time(y)
        y = F.silu(y)
        y = self.convlayer_time(y)

        return y + self.residual_layer(residue)


class SwitchSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, context, time):
        for layer in self.layers:
            if isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = nn.ModuleList(
            [
                UNET_ResidualBlock(1280, 1280),
                UNET_AttentionBlock(8, 160),
                UNET_ResidualBlock(1280, 1280),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(960, 320),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 320),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 320),
                    UNET_AttentionBlock(8, 80),
                ),
            ]
        )

    def forward(self, x, context, time):

        skip_connections = []
        for layer in self.encoder:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoder:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)

        return x


class UNET_Out(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.convlayer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.convlayer(x)

        return x


class Diffusion(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.time = TimeEmbedding(320)
        self.unet = UNET()
        self.output = UNET_Out(320, 4)

    def forward(self, x, time):

        time = self.time(time)

        out = self.unet(x)

        out = self.output(out)

        return out
