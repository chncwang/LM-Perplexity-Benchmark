import unittest

import torch
import torch.nn as nn

from lm_perplexity_benchmark.model import CustomLSTM


class TestCustomLSTM(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 20
        self.hippo_dim = 32
        self.batch_size = 5
        self.seq_length = 8
        self.model = CustomLSTM(self.input_dim, self.hidden_dim, self.hippo_dim)

    def test_initialization(self):
        """Test if model parameters are initialized correctly"""
        # Test dimensions of LSTM cell weights
        input_size = self.input_dim + self.hippo_dim  # LSTM input includes hippo state
        self.assertEqual(
            self.model.lstm_cell.weight_ih.shape, (4 * self.hidden_dim, input_size)
        )
        self.assertEqual(
            self.model.lstm_cell.weight_hh.shape, (4 * self.hidden_dim, self.hidden_dim)
        )

        # Test dimensions of hippo-related matrices
        self.assertEqual(
            self.model.hidden_to_hippo.weight.shape, (self.hippo_dim, self.hidden_dim)
        )
        self.assertEqual(self.model.W_hippo_i.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_f.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_g.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_o.shape, (self.hippo_dim, self.hidden_dim))

        # Test dimensions of special matrices
        self.assertEqual(self.model.A.shape, (self.hippo_dim, self.hippo_dim))
        self.assertEqual(self.model.B.shape, (self.hippo_dim, self.hippo_dim))

    def test_forward_shape(self):
        """Test if forward pass produces correct output shape"""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output, cell = self.model(x)

        expected_shape = (self.batch_size, self.seq_length, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(cell.shape, expected_shape)

    def test_forward_device_consistency(self):
        """Test if model handles device placement correctly"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = self.model.to(device)
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(device)
            output = model(x)
            self.assertEqual(output.device, device)

    def test_xavier_initialization(self):
        """Test if Xavier initialization is applied correctly"""
        # Test LSTM cell weights
        lstm_weights = [
            self.model.lstm_cell.weight_ih,  # input-to-hidden weights
            self.model.lstm_cell.weight_hh,  # hidden-to-hidden weights
        ]

        # Test Hippo-related weights
        hippo_weights = [
            self.model.hidden_to_hippo.weight,
            self.model.W_hippo_i,
            self.model.W_hippo_f,
            self.model.W_hippo_g,
            self.model.W_hippo_o,
        ]

        # Test all weights
        for param in lstm_weights + hippo_weights:
            # Check if weights are not all zero
            self.assertFalse(torch.all(param == 0))
            # Check if variance is within reasonable bounds for Xavier initialization
            var = torch.var(param).item()
            self.assertGreater(var, 0)
            self.assertLess(var, 1)

        # Test that biases are initialized to zero
        # Note: Removed bias initialization tests since LSTM cell biases are not initialized to zero by default
        self.assertTrue(torch.all(self.model.hidden_to_hippo.bias == 0))

    def test_sequence_independence(self):
        """Test if each sequence in batch is processed independently"""
        # Set model to eval mode to disable dropout
        self.model.eval()

        # Create two identical sequences in a batch
        x1 = torch.randn(1, self.seq_length, self.input_dim)
        x2 = torch.cat([x1, x1], dim=0)  # Create batch of two identical sequences

        with torch.no_grad():  # Disable gradient computation for inference
            output1, cell1 = self.model(x1)
            output2, cell2 = self.model(x2)

        # Check if the outputs for identical sequences are the same
        torch.testing.assert_close(output2[0], output2[1])
        torch.testing.assert_close(output1[0], output2[0])
        torch.testing.assert_close(cell1[0], cell2[1])
        torch.testing.assert_close(cell1[0], cell2[0])


if __name__ == "__main__":
    unittest.main()
