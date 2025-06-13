import math
import torch
from torch import nn


class InputEmbedding(nn.Module):
    """
    Input embeddings for the Transformer model.
    Converts input token IDs into dense vectors and scales them.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the InputEmbedding layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the embedding vectors.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Create an embedding layer that maps token IDs to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for input embeddings.

        Args:
            x (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded and scaled tensor (batch_size, seq_len, d_model).
        """
        # Multiply by sqrt(d_model) as per the transformer paper's scaling factor
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    Adds sinusoidal positional information to the input embeddings.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the embedding vectors.
            seq_len (int): The maximum sequence length.
            dropout (float): The dropout rate to apply to the positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a position vector (seq_len, 1) for calculating sin/cos arguments
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        # Calculate the division term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension: (1, seq_len, d_model)
        # Register 'pe' as a buffer, so it's part of the module's state but not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model), typically embeddings.

        Returns:
            torch.Tensor: Input tensor with positional encoding added and dropout applied.
        """
        # Add positional encoding to the input.
        # We slice pe to match the current sequence length of x.
        # .requires_grad_(False) ensures that positional encodings are not updated during training.
        x = x + (self.pe[:, : x.size(1)]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Layer Normalization module.
    Normalizes the input across the last dimension.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initializes the LayerNormalization layer.

        Args:
            features (int): The number of features (dimension size) to normalize over.
                            This typically corresponds to d_model in Transformer.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps

        # Initialize weights and bias to be learnable parameters, with shape matching features
        self.weights = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # Apply layer normalization formula
        return self.weights * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    """
    Feed-forward network module for the Transformer.
    Consists of two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initializes the FeedForward layer.

        Args:
            d_model (int): The dimensionality of the input and output vectors.
            d_ff (int): The dimensionality of the inner layer (feed-forward dimension).
            dropout (float): The dropout rate to apply.
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.layer2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward network.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """
        # Apply first linear layer, ReLU activation, dropout, and then second linear layer
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))


class MultiheadAttention(nn.Module):
    """docstring"""

    def __init__(self, d_model: int, n_heads: int, dropout: int) -> None:
        """docstring"""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        assert d_model % n_heads == 0

        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """docstring"""

        d_k = query.shape[-1]

        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, que, ke, val, mask):
        """docstring"""
        query = self.q(que)
        key = self.k(ke)
        value = self.v(val)

        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        x, self.attention_scores = MultiheadAttention.attention(
            query, key, value, mask, self.dropout
        )

        out = self.concat(x)
        output = self.output(out)

        return output

    def concat(self, tensor):
        """docstring"""
        batch_size, h, seq_len, d_k = tensor.size()
        d_model = h * d_k

        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)


class ResidualConnection(nn.Module):
    """docstring"""

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.features = LayerNormalization(features)

    def forward(self, x, sublayer):
        """docstring"""
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    """docstring"""

    def __init__(
        self,
        features: int,
        attention_block: MultiheadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, mask):
        """docstring"""
        x = self.residual[0](x, lambda x: self.self.attention_block(x, x, x, mask))
        x = self.residula[1](x, self.feed_forward)
        return x


class EncoderLayer(nn.Module):
    """docstring"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """docstring"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """docstring"""

    def __init__(
        self,
        features: int,
        masked_attention: MultiheadAttention,
        attention: MultiheadAttention,
        feedforward: FeedForward,
        dropout: float,
    ) -> None:
        """docstring"""
        super().__init__()
        self.masked_attention = masked_attention
        self.attention = attention
        self.feedforward = feedforward
        self.residual_connection = nn.ModuleList(
            ResidualConnection(features, dropout) for _ in range(3)
        )

    def forward(self, x, encoder_output, mask1, mask2):
        """docstring"""
        x = self.residual_connection[0](
            x, lambda x: self.masked_attention(x, x, x, mask1)
        )
        x = self.residual_connection[1](
            x, lambda x: self.attention(x, encoder_output, encoder_output, mask2)
        )
        x = self.residual_connection[2](x, lambda x: self.feedforward)
        return x


class DecoderLayer(nn.Module):
    """docstring"""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, mask1, mask2):
        """docstring"""
        for layer in self.layers:
            x = layer(x, encoder_output, mask1, mask2)
        return self.norm(x)


class Projection(nn.Module):
    """docstring"""

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """docstring"""
        return self.projection(x)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: EncoderLayer,
        decoder: DecoderLayer,
        input_emd: InputEmbedding,
        output_embedding: InputEmbedding,
        input_pos: PositionalEncoding,
        output_pos: PositionalEncoding,
        projection_layer: Projection,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.input_emd = input_emd
        self.output_embedding = output_embedding
        self.input_pos = input_pos
        self.output_pos = output_pos
        self.projection_layer = projection_layer

    def encode(self, source, mask):
        """docstring"""
        source = self.input_emd(source)
        source = self.input_pos(source)
        return self.encoder(source, mask)

    def decode(self, encoder_output, input_mask, output, output_mask):
        """docstring"""
        output = self.output_embedding(output)
        output = self.output_pos(output)
        return self.decoder(output, encoder_output, input_mask, output_mask)


def transformer_implement(
    input_vocab_size: int,
    output_vocab_size: int,
    input_seq_len: int,
    output_seq_len: int,
    d_model: int = 512,
    n: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> None:
    input_emd = InputEmbedding(d_model, input_vocab_size)
    output_embedding = InputEmbedding(d_model, output_vocab_size)

    input_pos = PositionalEncoding(d_model, input_seq_len, dropout)
    output_pos = PositionalEncoding(d_model, output_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n):
        encoder_attention = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = Encoder(d_model, encoder_attention, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(n):
        masked_attention = MultiheadAttention(d_model, h, dropout)
        attention = MultiheadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = Decoder(
            d_model, masked_attention, attention, feed_forward, dropout
        )
        decoder_blocks.append(decoder_block)

    encoder = EncoderLayer(d_model, nn.ModuleList(encoder_blocks))
    decoder = DecoderLayer(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = Projection(d_model, output_vocab_size)

    transformer = Transformer(
        encoder,
        decoder,
        input_emd,
        output_embedding,
        input_pos,
        output_pos,
        projection_layer,
    )

    for p in transformer.parameter():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
