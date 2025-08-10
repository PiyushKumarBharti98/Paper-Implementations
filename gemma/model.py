import math
import torch.nn.functional as F
from torch import nn, rms_norm
import torch


class InputEmbedding(nn.Module):
    """docstring"""

    def __init__(self, vocab_size: int, d_model: int):
        """docstring"""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """docstring"""
        return self.embedding(x) * math.sqrt(self.d_model)


class RoPositionalEncoding(nn.Module):
    """docstring"""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(seq_len, dtype=torch.float)
        div_term = 1.0 / (
            10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        )

        freq = position.unsqueeze(1) * div_term.unsqueeze(0)

        pe[:, 0::2] = torch.sin(freq)
        pe[:, 1::2] = torch.cos(freq)

        pe = pe.unsqueeze(0)  # Add batch dimension: (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """docstring"""
        pos_emb = self.pe[:, : x.size(1), :].requires_grad_(False)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        sin = pos_emb[..., 0::2]
        cos = pos_emb[..., 1::2]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * sin + x2 * cos

        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(start_dim=-2)

        return self.dropout(rotated_x)


class RMSNorm(nn.Module):
    """docstring"""

    def __init__(self, dim, eps: float = 1e-10) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """docstring"""
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class MultiQueryAttention(nn.Module):
    """docstring"""

    def __init__(self, n_heads: int, d_model, dropout: float) -> None:
        """docstring"""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0

        self.d_k = self.d_model // self.n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, self.d_k)
        self.v = nn.Linear(d_model, self.d_k)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        """docstring"""
        batch_size, seq_len, _ = x.shape

        q = self.q(x)
        q = q.view(batch_size, seq_len, self.d_k, self.n_heads).transpose(1, 2)

        k = self.k(x)
        v = self.v(x)

        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output(out)


class FeedForwardNetwork(nn.Module):
    """docstring"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """docstring"""
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """docstring"""
        return self.layer2(self.dropout(nn.GELU(self.layer1(x))))


class Decoder(nn.Module):
    """docstring"""

    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float) -> None:
        """docstring"""
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attention_network = MultiQueryAttention(n_heads, d_model, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x):
        """docstring"""
        residual = x
        x_norm = self.attention_norm(x)
        attention_output = self.attention_network(x_norm)
        x = residual + attention_output

        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x


class GemmaModel(nn.Module):
    """docstring"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        seq_len: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_layers: int,
    ) -> None:
        """docstring"""
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.d_model = d_model
        self.rope = RoPositionalEncoding(seq_len, d_model, dropout)
        self.decoder_block = nn.ModuleList(
            [Decoder(n_heads, d_model, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.embedding.weight = self.output_layer.weight

    def forward(self, x, mask=None):
        """docstring"""
        x = self.embedding(x) * math.sqrt(self.d_model)
        for block in self.decoder_block:
            x = block(x, self.rope, mask)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits


if __name__ == "__main__":
    VOCAB_SIZE = 10000
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    SEQ_LEN = 128
    BATCH_SIZE = 4

    causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).view(1, 1, SEQ_LEN, SEQ_LEN)

    print("Instantiating Gemma model...")
    model = GemmaModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
    )
    print("Model instantiated successfully.")
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("model----/n")
    print(model)
