import math
import torch
from torch import nn


class InputEmbedding(nn.Module):
    """docstring"""

    def __init__(self, vocab_size, d_model):
        """docstring"""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """docstring"""
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionaEncoding(nn.Module):
    """docstring"""

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 0::1] = torch.cos(pos * div_term)

        self.pe = self.pe.unsqueeze(1)
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        """docstring"""
        x = x + (self.pe[:, : x.size(1)]).requires_grad_(False)
        return self.dropout(x)


class LinearNormalization(nn.Module):
    """docstring"""

    def __init__(self, features, eps=1e-10):
        """docstring"""
        super().__init__()
        self.eps = eps

        self.weights = nn.Parameter(torch.zeros(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """docstring"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.weights * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    """docstring"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """docstring"""
        return self.layer2(self.dropout(self.layer1(x)))


class MultiHeadAttention(nn.Module):
    """docstring"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """docstring"""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads

        self.q_weights = nn.Linear(d_model, d_model, bias=True)
        self.k_weights = nn.Linear(d_model, d_model, bias=True)
        self.v_weights = nn.Linear(d_model, d_model, bias=True)
        self.o_weights = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """docstring"""
        d_k = query.shape[-1]

        attention_scores = (query @ key.transporse(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """docstring"""
        query = self.q_weights(q)
        key = self.k_weights(k)
        value = self.v_weights(v)

        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transporse(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transporse(
            1, 2
        )

        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transporse(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        x = (
            x.transporse(1, 2)
            .contigious()
            .view(x.shape[0], -1, self.n_heads * self.d_k)
        )

        return self.o_weights(x)


class Encoder(nn.Module):
    """docstring"""

    def __init__(
        self,
    ) -> None:
        """docstring"""
        super().__init__()

    def forward(self):
        """docstring"""
