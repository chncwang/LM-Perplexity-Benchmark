from typing import Optional

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A LSTM language model.
    """

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialize the model.

        @param input_size: The size of the input.
        @param embedding_size: The size of the embedding.
        @param hidden_size: The size of the hidden state.
        @param num_layers: The number of layers.
        @param dropout: The dropout probability.
        """
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, input_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model.
        """
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        @param x: The input tensor.
        @param hidden: The hidden state.
        @return: The output tensor and the hidden state.
        """
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the hidden state of the model.

        @param batch_size: The batch size.
        @param device: The device.
        @return: The hidden state.
        """
        weight = next(self.parameters())
        hidden_size = self.lstm.hidden_size
        num_layers = self.lstm.num_layers
        return (
            weight.new_zeros(num_layers, batch_size, hidden_size).to(device),
            weight.new_zeros(num_layers, batch_size, hidden_size).to(device),
        )
