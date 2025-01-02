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
        # Test dimensions of weight matrices
        self.assertEqual(self.model.W_ii.shape, (self.input_dim, self.hidden_dim))
        self.assertEqual(self.model.W_if.shape, (self.input_dim, self.hidden_dim))
        self.assertEqual(self.model.W_ig.shape, (self.input_dim, self.hidden_dim))
        self.assertEqual(self.model.W_io.shape, (self.input_dim, self.hidden_dim))

        # Test dimensions of hidden-to-hidden matrices
        self.assertEqual(self.model.W_hi.shape, (self.hidden_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hf.shape, (self.hidden_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hg.shape, (self.hidden_dim, self.hidden_dim))
        self.assertEqual(self.model.W_ho.shape, (self.hidden_dim, self.hidden_dim))

        # Test dimensions of hippo-related matrices
        self.assertEqual(self.model.W_hippo_i.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_f.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_g.shape, (self.hippo_dim, self.hidden_dim))
        self.assertEqual(self.model.W_hippo_o.shape, (self.hippo_dim, self.hidden_dim))

        # Test dimensions of bias vectors
        self.assertEqual(self.model.b_i.shape, (self.hidden_dim,))
        self.assertEqual(self.model.b_f.shape, (self.hidden_dim,))
        self.assertEqual(self.model.b_g.shape, (self.hidden_dim,))
        self.assertEqual(self.model.b_o.shape, (self.hidden_dim,))

        # Test dimensions of special matrices
        self.assertEqual(self.model.A.shape, (self.hippo_dim, self.hippo_dim))
        self.assertEqual(self.model.B.shape, (self.hippo_dim, self.hippo_dim))

    def test_forward_shape(self):
        """Test if forward pass produces correct output shape"""
        x = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output, _ = self.model(x)

        expected_shape = (self.batch_size, self.seq_length, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_device_consistency(self):
        """Test if model handles device placement correctly"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = self.model.to(device)
            x = torch.randn(self.batch_size, self.seq_length, self.input_dim).to(device)
            output, _ = model(x)
            self.assertEqual(output.device, device)

    def test_xavier_initialization(self):
        """Test if Xavier initialization is applied correctly"""
        # Test if weights are not zero and have reasonable variance
        for param in [
            self.model.W_ii,
            self.model.W_if,
            self.model.W_ig,
            self.model.W_io,
            self.model.W_hi,
            self.model.W_hf,
            self.model.W_hg,
            self.model.W_ho,
            self.model.W_hippo_i,
            self.model.W_hippo_f,
            self.model.W_hippo_g,
            self.model.W_hippo_o,
        ]:
            # Check if weights are not all zero
            self.assertFalse(torch.all(param == 0))
            # Check if variance is within reasonable bounds for Xavier initialization
            var = torch.var(param).item()
            self.assertGreater(var, 0)
            self.assertLess(var, 1)

    def test_sequence_independence(self):
        """Test if each sequence in batch is processed independently"""
        # Create two identical sequences in a batch
        x1 = torch.randn(1, self.seq_length, self.input_dim)
        x2 = torch.cat([x1, x1], dim=0)  # Create batch of two identical sequences

        output1, _ = self.model(x1)
        output2, _ = self.model(x2)

        # Check if the outputs for identical sequences are the same
        torch.testing.assert_close(output2[0], output2[1])
        torch.testing.assert_close(output1[0], output2[0])


if __name__ == "__main__":
    unittest.main()
