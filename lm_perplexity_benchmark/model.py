from typing import Optional

import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights for input-to-hidden connections
        self.W_ii = nn.Parameter(torch.randn(input_dim, hidden_dim))  # input gate
        self.W_if = nn.Parameter(torch.randn(input_dim, hidden_dim))  # forget gate
        self.W_ig = nn.Parameter(torch.randn(input_dim, hidden_dim))  # cell gate
        self.W_io = nn.Parameter(torch.randn(input_dim, hidden_dim))  # output gate

        # Initialize weights for hidden-to-hidden connections
        self.W_hi = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # input gate
        self.W_hf = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # forget gate
        self.W_hg = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # cell gate
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, hidden_dim))  # output gate

        # Initialize Hippo-related weights
        self.W_hippo_i = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_hippo_f = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_hippo_g = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_hippo_o = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Initialize biases
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))  # input gate
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))  # forget gate
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))  # cell gate
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))  # output gate

        # Use cached A and B matrices if they exist for this hidden dimension
        if not hasattr(CustomLSTM, "_cached_matrices"):
            CustomLSTM._cached_matrices = {}

        if hidden_dim not in CustomLSTM._cached_matrices:
            # Initialize special weight matrices A and B more efficiently
            n_indices = torch.arange(hidden_dim, dtype=torch.float32)
            k_indices = torch.arange(hidden_dim, dtype=torch.float32)

            # Initialize B more efficiently
            B = torch.sqrt(2 * n_indices + 1).unsqueeze(1).expand(-1, hidden_dim)

            # Initialize A more efficiently
            A = torch.zeros(hidden_dim, hidden_dim)
            n_expanded = (2 * n_indices + 1).unsqueeze(1)
            k_expanded = (2 * k_indices + 1).unsqueeze(0)
            mask = torch.tril(torch.ones(hidden_dim, hidden_dim))
            A = torch.where(
                mask > 0,
                torch.where(
                    torch.eye(hidden_dim) > 0,
                    n_indices + 1,
                    torch.sqrt(n_expanded * k_expanded),
                ),
                torch.zeros(hidden_dim, hidden_dim),
            )
            CustomLSTM._cached_matrices[hidden_dim] = (A, B)
        else:
            A, B = CustomLSTM._cached_matrices[hidden_dim]

        # Register A and B as buffers
        self.register_buffer("A", A)
        self.register_buffer("B", B)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for param in [
            self.W_ii,
            self.W_if,
            self.W_ig,
            self.W_io,
            self.W_hi,
            self.W_hf,
            self.W_hg,
            self.W_ho,
            self.W_hippo_i,
            self.W_hippo_f,
            self.W_hippo_g,
            self.W_hippo_o,
        ]:
            nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        Forward pass of the LSTM
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            outputs: Output tensor of shape (batch_size, sequence_length, hidden_dim)
            cell_states: Cell state tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        batch_size, seq_length, _ = x.size()

        # Initialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        # hippo_c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # Container for output sequences and cell states
        outputs = []
        cell_states = []

        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]  # Current input (batch_size, input_dim)

            # hippo_c_t = (1.0 - self.A / (t + 1)) * hippo_c_t + 1.0 / (
            #     t + 1.0
            # ) * self.B * h_t

            # Input gate
            i_t = torch.sigmoid(
                x_t @ self.W_ii
                # + hippo_c_t @ self.W_hippo_i
                + h_t @ self.W_hi
                + self.b_i
            )

            # Forget gate
            f_t = torch.sigmoid(
                x_t @ self.W_if
                # + hippo_c_t @ self.W_hippo_f
                + h_t @ self.W_hf
                + self.b_f
            )

            # Cell gate (candidate)
            g_t = torch.tanh(
                x_t @ self.W_ig
                # + hippo_c_t @ self.W_hippo_g
                + h_t @ self.W_hg
                + self.b_g
            )

            # Output gate
            o_t = torch.sigmoid(
                x_t @ self.W_io
                # + hippo_c_t @ self.W_hippo_o
                + h_t @ self.W_ho
                + self.b_o
            )

            # Update cell state
            c_t = f_t * c_t + i_t * g_t

            # Update hidden state
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t)
            cell_states.append(c_t)

        # Stack outputs and cell states along sequence dimension
        outputs = torch.stack(
            outputs, dim=1
        )  # (batch_size, sequence_length, hidden_dim)
        cell_states = torch.stack(
            cell_states, dim=1
        )  # (batch_size, sequence_length, hidden_dim)

        return outputs, cell_states


class ResidualLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        lstm_class: Optional[type] = nn.LSTM,
    ) -> None:
        """
        Initialize the residual LSTM layer.
        @param input_size: Size of the input features
        @param hidden_size: Size of the hidden state
        @param dropout: Dropout probability
        @param lstm_class: LSTM class to use (defaults to nn.LSTM)
        """
        super().__init__()

        # Create LSTM instance based on the provided class
        if lstm_class == nn.LSTM:
            self.lstm = lstm_class(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif lstm_class == CustomLSTM:
            self.lstm = lstm_class(input_dim=input_size, hidden_dim=hidden_size)
        else:
            raise ValueError(f"Unsupported LSTM class: {lstm_class}")

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
        lstm_class: Optional[type] = nn.LSTM,
    ) -> None:
        """
        Initialize the model.
        @param vocab_size: Size of the vocabulary
        @param embedding_size: Size of the embeddings
        @param hidden_size: Size of the hidden state
        @param num_layers: Number of stacked LSTM layers
        @param dropout: Dropout probability
        @param lstm_class: LSTM class to use (defaults to nn.LSTM)
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.embedding_dropout = nn.Dropout(dropout)

        # First LSTM layer processes embeddings
        self.first_layer = ResidualLSTMLayer(
            input_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            lstm_class=lstm_class,
        )

        # Additional layers process hidden states
        self.layers = nn.ModuleList(
            [
                ResidualLSTMLayer(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    lstm_class=lstm_class,
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
