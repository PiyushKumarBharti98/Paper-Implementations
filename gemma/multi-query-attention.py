import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    """docstring"""

    def __init__(self, n_heads: int, d_model, dropout: float, n_query: int) -> None:
        """docstring"""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_query = n_query

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

        attention_scores = (q * k.transpose(-2, -1)) / (self.d_k**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = attention_weights * v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output(out)
