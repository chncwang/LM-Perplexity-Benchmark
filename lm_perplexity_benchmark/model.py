from typing import Optional

import torch
import torch.nn as nn


class ResidualLSTMLayer(nn.Module):
    """
    A single LSTM layer with residual connection.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.5) -> None:
        """
        Initialize the residual LSTM layer.
        @param input_size: Size of the input features
        @param hidden_size: Size of the hidden state
        @param dropout: Dropout probability
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Add a projection layer if input and hidden sizes differ
        self.projection = None
        if input_size != hidden_size:
            self.projection = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        @param x: Input tensor of shape (batch_size, sequence_length, input_size)
        @return: Output tensor of shape (batch_size, sequence_length, hidden_size)
        """
        residual = x if self.projection is None else self.projection(x)
        output, _ = self.lstm(x)
        output = self.dropout(output)
        return output + residual


class LSTMModel(nn.Module):
    """
    A sequence-to-sequence LSTM language model with residual connections.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the model.
        @param vocab_size: Size of the vocabulary
        @param embedding_size: Size of the embeddings
        @param hidden_size: Size of the hidden state
        @param num_layers: Number of stacked LSTM layers
        @param dropout: Dropout probability
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.embedding_dropout = nn.Dropout(dropout)

        # First LSTM layer processes embeddings
        self.first_layer = ResidualLSTMLayer(
            input_size=embedding_size, hidden_size=hidden_size, dropout=dropout
        )

        # Additional layers process hidden states
        self.layers = nn.ModuleList(
            [
                ResidualLSTMLayer(
                    input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
                )
                for _ in range(num_layers - 1)
            ]
        )

        # If the embedding size is different from the hidden size, add a projection layer
        if embedding_size != hidden_size:
            self.projection = nn.Linear(hidden_size, embedding_size)
        else:
            self.projection = None

        self.output = nn.Linear(hidden_size, vocab_size)

        # Tie weights between input embedding and output embedding
        self.output.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.
        """
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        @param x: Input tensor of shape (batch_size, sequence_length)
        @return: Output tensor of shape (batch_size, sequence_length, vocab_size)
        """
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_size)
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        # Process through first LSTM layer
        x = self.first_layer(x)

        # Process through additional layers
        for layer in self.layers:
            x = layer(x)

        # Project to hidden size if necessary
        if self.projection is not None:
            x = self.projection(x)

        # Project to vocabulary size
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, vocab_size)
        logits = self.output(x)

        return logits
