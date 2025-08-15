import math
from typing import final
import torch
from torch import batch_norm, embedding, nn


class PatchEmbeddings(nn.Module):
    """docstring"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embedding_size: int = 768,
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
