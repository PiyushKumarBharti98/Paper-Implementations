import math
from typing_extensions import final
import torch
from torch import imag, nn


def image_to_path(image, patch_size):
    """docstring"""
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(1, 3, -1, patch_size * patch_size)
    patches = patches.permute(2, 3, 1)
    return patches


class Embeddings(nn.Module):
    """docstring"""

    def __init__(self, image_size: int, patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Linear(image_size * patch_size * patch_size, embedding_dim)

        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.position_embedding = nn.Parameter(
            torch.rand(1, self.num_patches + 1, embedding_dim)
        )

    def forward(self, image):
        """docstring"""

        batch_size, channels, height, width = image.size
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        patches = patches.contiguous().view(batch_size, self.num_patches, -1)

        patch_embeddings = self.projection(patches)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)

        final_embeddings = embeddings + self.position_embedding

        return final_embeddings


class LayerNorm(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """docstring"""
        return x


class MLP(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """docstring"""
        return x


class FeedForwardNetwork(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """docstring"""
        return x


class MultiHeadAttention(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """docstring"""
        return x


class ViTModel(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """docstring"""
        return x
