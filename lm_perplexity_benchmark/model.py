import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hippo_dim: int):
        """
        Initialize the CustomLSTM model using PyTorch's LSTMCell while maintaining Hippo state.
        @param input_dim: Dimension of the input features
        @param hidden_dim: Dimension of the hidden state
        @param hippo_dim: Dimension of the hippo state
        """
        super(CustomLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hippo_dim = hippo_dim

        # Use the standard PyTorch LSTMCell for the LSTM portion
        self.lstm_cell = nn.LSTMCell(input_dim + hippo_dim, hidden_dim)

        # Initialize Hippo-related weights
        self.hidden_to_hippo = nn.Linear(hidden_dim, hippo_dim)
        self.W_hippo_i = nn.Parameter(torch.randn(hippo_dim, hidden_dim))
        self.W_hippo_f = nn.Parameter(torch.randn(hippo_dim, hidden_dim))
        self.W_hippo_g = nn.Parameter(torch.randn(hippo_dim, hidden_dim))
        self.W_hippo_o = nn.Parameter(torch.randn(hippo_dim, hidden_dim))

        # Use cached A and B matrices if they exist for this hippo dimension
        if not hasattr(CustomLSTM, "_cached_matrices"):
            CustomLSTM._cached_matrices = {}

        if hippo_dim not in CustomLSTM._cached_matrices:
            # Initialize special weight matrices A and B
            n_indices = torch.arange(hippo_dim, dtype=torch.float32)
            k_indices = torch.arange(hippo_dim, dtype=torch.float32)

            # Create B tensor with proper memory allocation
            B = (
                torch.sqrt(2 * n_indices + 1).unsqueeze(1).expand(-1, hippo_dim).clone()
                * 0.5
            )

            A = torch.zeros(hippo_dim, hippo_dim)
            n_expanded = (2 * n_indices + 1).unsqueeze(1)
            k_expanded = (2 * k_indices + 1).unsqueeze(0)
            mask = torch.tril(torch.ones(hippo_dim, hippo_dim))
            A = torch.where(
                mask > 0,
                torch.where(
                    torch.eye(hippo_dim) > 0,
                    n_indices + 1,
                    torch.sqrt(n_expanded * k_expanded),
                ),
                torch.zeros(hippo_dim, hippo_dim),
            )
            A *= 0.5
            CustomLSTM._cached_matrices[hippo_dim] = (A, B)
        else:
            A, B = CustomLSTM._cached_matrices[hippo_dim]

        # Register A and B as buffers so they are moved correctly on .to(device) calls
        self.register_buffer("A", A)
        self.register_buffer("B", B)

        logger.debug(f"CustomLSTM.__init__: A: {A}")
        logger.debug(f"CustomLSTM.__init__: B: {B}")

        # Initialize Hippo-related weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize Hippo-related weights using Xavier initialization."""
        # Initialize weights (2D tensors) with xavier_uniform
        for param in [
            self.hidden_to_hippo.weight,
            self.W_hippo_i,
            self.W_hippo_f,
            self.W_hippo_g,
            self.W_hippo_o,
        ]:
            nn.init.xavier_uniform_(param)

        # Initialize bias (1D tensor) with zeros
        nn.init.zeros_(self.hidden_to_hippo.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM with Hippo integration.
        @param x: Input tensor of shape (batch_size, sequence_length, input_dim)
        @return:
            outputs: Output tensor of shape (batch_size, sequence_length, hidden_dim)
            cell_states: Cell state tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        batch_size, seq_length, _ = x.size()

        # Initialize states
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        hippo_c_t = torch.zeros(batch_size, self.hippo_dim, device=x.device)

        outputs = []
        cell_states = []

        for t in range(seq_length):
            x_t = x[:, t, :]  # (batch_size, input_dim)
            logger.debug(f"CustomLSTM.forward: x_t: {x_t} shape: {x_t.shape}")

            # Run standard LSTMCell with proper dimensions
            lstm_input = torch.cat(
                [x_t, hippo_c_t], dim=1
            )  # Changed parentheses to square brackets
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))
            logger.debug(f"CustomLSTM.forward: h_t: {h_t} shape: {h_t.shape}")
            logger.debug(f"CustomLSTM.forward: c_t: {c_t} shape: {c_t.shape}")

            # Add numerical stability to scaling factors
            max_scale = 1000.0  # Tune this value based on your needs

            scaling_factor_A = 1.0 - self.A / (t + 1.0)
            scaling_factor_A = torch.clamp(
                scaling_factor_A, min=-max_scale, max=max_scale
            )
            scaling_factor_A = scaling_factor_A.unsqueeze(0).expand(batch_size, -1, -1)

            scaling_factor_B = self.B / (t + 1.0)
            scaling_factor_B = torch.clamp(
                scaling_factor_B, min=-max_scale, max=max_scale
            )
            scaling_factor_B = scaling_factor_B.unsqueeze(0).expand(batch_size, -1, -1)

            logger.debug(
                f"CustomLSTM.forward: scaling_factor_A: {scaling_factor_A} shape: {scaling_factor_A.shape}"
            )
            logger.debug(
                f"CustomLSTM.forward: scaling_factor_B: {scaling_factor_B} shape: {scaling_factor_B.shape}"
            )

            # Project hidden state with stability
            f_t = self.hidden_to_hippo(h_t)  # (batch_size, hippo_dim)
            f_t = torch.clamp(f_t, min=-max_scale, max=max_scale)
            logger.debug(f"CustomLSTM.forward: f_t: {f_t} shape: {f_t.shape}")

            # Update hippo cell state with stability checks
            hippo_c_t = (scaling_factor_A.squeeze(2) @ hippo_c_t.unsqueeze(2)).squeeze(
                2
            )
            hippo_c_t = hippo_c_t + (
                scaling_factor_B.squeeze(2) @ f_t.unsqueeze(2)
            ).squeeze(2)
            hippo_c_t = torch.clamp(hippo_c_t, min=-max_scale, max=max_scale)
            logger.debug(
                f"CustomLSTM.forward: hippo_c_t: {hippo_c_t} shape: {hippo_c_t.shape}"
            )

            outputs.append(h_t)
            cell_states.append(c_t)

        # Stack outputs along the sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_length, hidden_dim)
        cell_states = torch.stack(
            cell_states, dim=1
        )  # (batch_size, seq_length, hidden_dim)

        return outputs, cell_states


class ResidualLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        lstm_class: Optional[type] = nn.LSTM,
        hippo_dim: int = 32,
    ) -> None:
        """
        Initialize the residual LSTM layer.
        @param input_size: Size of the input features
        @param hidden_size: Size of the hidden state
        @param dropout: Dropout probability
        @param lstm_class: LSTM class to use (defaults to nn.LSTM)
        @param hippo_dim: Dimension of the hippo state
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
            self.lstm = lstm_class(
                input_dim=input_size, hidden_dim=hidden_size, hippo_dim=hippo_dim
            )
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
        logger.debug(
            f"ResidualLSTMLayer.forward: residual: {residual} shape: {residual.shape}"
        )
        output, _ = self.lstm(x)
        logger.debug(
            f"ResidualLSTMLayer.forward: output: {output} shape: {output.shape}"
        )
        output = self.dropout(output)
        logger.debug(
            f"ResidualLSTMLayer.forward: output after dropout: {output} shape: {output.shape}"
        )
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
        hippo_dim: int = 32,
    ) -> None:
        """
        Initialize the model.
        @param vocab_size: Size of the vocabulary
        @param embedding_size: Size of the embeddings
        @param hidden_size: Size of the hidden state
        @param num_layers: Number of stacked LSTM layers
        @param dropout: Dropout probability
        @param lstm_class: LSTM class to use (defaults to nn.LSTM)
        @param hippo_dim: Dimension of the hippo state
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
        logger.debug(
            f"LSTMModel.forward: x after embedding dropout: {x} shape: {x.shape}"
        )

        # Process through first LSTM layer
        x = self.first_layer(x)
        logger.debug(f"LSTMModel.forward: x after first layer: {x} shape: {x.shape}")
        # Process through additional layers
        for layer in self.layers:
            x = layer(x)
            logger.debug(f"LSTMModel.forward: x after layer: {x} shape: {x.shape}")

        # Project to hidden size if necessary
        if self.projection is not None:
            x = self.projection(x)
        logger.debug(f"LSTMModel.forward: x after projection: {x} shape: {x.shape}")

        # Project to vocabulary size
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, vocab_size)
        logits = self.output(x)
        logger.debug(f"LSTMModel.forward: logits: {logits} shape: {logits.shape}")

        return logits
