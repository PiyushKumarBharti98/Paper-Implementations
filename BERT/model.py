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


class PositionalEncoding(nn.Module):
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

        self.weights = nn.Parameter(torch.ones(features))
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
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))


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

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
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
        ).transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )

        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contigious().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.o_weights(x)


class ResidualConnection(nn.Module):
    """docstring"""

    def __init__(self, dropout: float, features: int) -> None:
        """docstring"""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LinearNormalization(features)

    def forward(self, x, sublayer):
        """docstring"""
        return x + self.layernorm(sublayer(self.dropout(x)))


class Encoder(nn.Module):
    """docstring"""

    def __init__(
        self,
        features: int,
        attention_block: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        """docstring"""
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList(
            [ResidualConnection(dropout, features) for _ in range(2)]
        )

    def forward(self, x, mask):
        """docstring"""
        x = self.residual[0](x, lambda x: self.attention_block(x, x, x, mask))
        x = self.residual[1](x, self.feed_forward)
        return x


class EncoderLayer(nn.Module):
    """docstring"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """docstring"""
        super().__init__()
        self.layers = layers
        self.norm = LinearNormalization(features)

    def forward(self, x, mask):
        """docstring"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class BERT(nn.Module):
    """BERT model implementation"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 512,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_len, d_model, dropout)

        encoder_blocks = []
        for _ in range(n_layers):
            attention = MultiHeadAttention(d_model, n_heads, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout)
            encoder_block = Encoder(d_model, attention, feed_forward, dropout)
            encoder_blocks.append(encoder_block)

        self.encoder = EncoderLayer(d_model, nn.ModuleList(encoder_blocks))

    def forward(self, x, mask=None):
        """forward"""
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        return x
