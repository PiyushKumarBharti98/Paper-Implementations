import torch
from torch import nn
from config import VIT_BASE_CONFIG as config


class PatchEmbeddings(nn.Module):
    """docstring"""

    def __init__(
        self,
        image_size: int = config["image_size"],
        patch_size: int = config["patch_size"],
        in_channels: int = config["in_channels"],
        embedding_size: int = config["embedding_size"],
    ) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embedding_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_size))
        self.patch_embeddings = nn.Parameter(
            torch.rand(1, self.num_patches + 1, embedding_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """docstring"""
        batch_size = image.shape[0]
        image = self.projection(image)
        image = image.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, image), dim=1)
        final_embeddings = embeddings + self.patch_embeddings
        return final_embeddings


class MultiHeadSelfAttention(nn.Module):
    """docstring"""

    def __init__(
        self,
        embedding_dim: int = config["embedding_dim"],
        num_heads: int = config["num_heads"],
        dropout: float = config["dropout"],
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim / num_heads
        assert self.num_heads * self.head_dim == embedding_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        B, N, D = x.shape

        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 2, 4, 1)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention = q @ k.transpose(-2, -1) * self.scale
        attention = attention.softmax(attention)
        attention = self.dropout(attention)

        weighted_average = attention @ v

        weighted_average = weighted_average.transpose(1, 2).reshape(B, N, D)

        out = self.out_proj(weighted_average)

        return out


class MLP(nn.Module):
    """docstring"""

    def __init__(
        self,
        embedding_size: int = config["embedding_size"],
        mlp_size: int = config["mlp_size"],
        dropout: float = config["dropout"],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        return self.net(x)


class VITEncoder(nn.Module):
    """docstring"""

    def __init__(
        self,
        embedding_size: int = config["embedding_size"],
        num_heads: int = config["num_heads"],
        mlp_size: int = config["mlp_size"],
        dropout: float = config["dropout"],
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadSelfAttention(embedding_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = MLP(embedding_size, mlp_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        x += self.attention(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class FinalModel(nn.Module):
    """docstring"""

    def __init__(
        self,
        image_size: int = config["image_size"],
        patch_size: int = config["patch_size"],
        in_channels: int = config["in_channels"],
        embedding_size: int = config["embedding_size"],
        num_heads: int = config["num_heads"],
        mlp_size: int = config["mlp_size"],
        dropout: float = config["dropout"],
        num_layers: int = config["num_layers"],
        num_classes: int = config["num_classes"],
    ) -> None:
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, in_channels, embedding_size
        )
        self.encoder = nn.Sequential(
            *[
                VITEncoder(embedding_size, num_heads, mlp_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        x = self.patch_embeddings(x)
        x = self.encoder(x)
        cls_token_output = x[:, 0]
        cls_token_output = self.norm(cls_token_output)

        logits = self.classifier(cls_token_output)

        return logits
