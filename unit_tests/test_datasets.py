import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from lm_perplexity_benchmark.datasets import RnnDataset


class TestRnnDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across test methods."""
        # Initialize a simple tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # Create sample texts
        texts = [
            "This is a short sequence.",
            "This is a longer sequence that should be longer than the short sequence.",
            "A",  # Test minimal sequence
        ]

        # Create a HuggingFace dataset
        cls.dataset = Dataset.from_dict({"text": texts})
        # Tokenize the texts
        cls.tokenized_dataset = cls.dataset.map(
            lambda x: cls.tokenizer(x["text"], truncation=True), remove_columns=["text"]
        )

    def test_initialization(self):
        """Test if the dataset initializes properly"""
        max_length = 512
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        self.assertEqual(len(dataset), len(self.tokenized_dataset))
        self.assertEqual(dataset.max_length, max_length)
        self.assertEqual(dataset.pad_token_id, self.tokenizer.pad_token_id)

    def test_getitem_dimensions(self):
        """Test if __getitem__ returns tensors of correct dimensions"""
        max_length = 10
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        input_ids, target_ids, mask = dataset[0]

        # Check dimensions
        self.assertEqual(input_ids.shape, torch.Size([max_length - 1]))
        self.assertEqual(target_ids.shape, torch.Size([max_length - 1]))
        self.assertEqual(mask.shape, torch.Size([max_length - 1]))

    def test_sequence_truncation(self):
        """Test if long sequences are properly truncated"""
        max_length = 5
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        # Get a long sequence
        input_ids, target_ids, mask = dataset[1]

        # Check if sequences are truncated
        self.assertEqual(len(input_ids), max_length - 1)
        self.assertEqual(len(target_ids), max_length - 1)
        self.assertEqual(len(mask), max_length - 1)

    def test_sequence_padding(self):
        """Test if short sequences are properly padded"""
        max_length = 10
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        # Get the shortest sequence (index 2: "A")
        input_ids, target_ids, mask = dataset[2]

        # Check padding
        self.assertTrue(torch.any(input_ids == dataset.pad_token_id))
        self.assertTrue(torch.any(target_ids == dataset.pad_token_id))
        # Check mask for padding
        self.assertTrue(torch.any(mask == False))  # Should have padding positions

    def test_input_target_relationship(self):
        """Test if input and target sequences are properly shifted"""
        max_length = 512
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        # Get original token sequence
        original_tokens = self.tokenized_dataset[0]["input_ids"]
        input_ids, target_ids, _ = dataset[0]

        # Check if input is original[:-1] and target is original[1:]
        # But only for the non-padded part
        valid_length = min(len(original_tokens) - 1, max_length - 1)
        self.assertTrue(
            torch.equal(
                input_ids[:valid_length],
                torch.tensor(original_tokens[:valid_length], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                target_ids[:valid_length],
                torch.tensor(original_tokens[1 : valid_length + 1], dtype=torch.long),
            )
        )

    def test_mask_correctness(self):
        """Test if masks correctly identify real tokens vs padding"""
        max_length = 10
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, max_length)

        # Test with a short sequence that will need padding
        input_ids, _, mask = dataset[2]

        # Count non-padding tokens in input
        real_tokens = torch.sum(input_ids != dataset.pad_token_id)
        # Count True values in mask
        mask_sum = torch.sum(mask)

        # The numbers should match
        self.assertEqual(real_tokens, mask_sum)

    def test_dtype_correctness(self):
        """Test if returned tensors have correct dtypes"""
        dataset = RnnDataset(self.tokenized_dataset, self.tokenizer, 512)
        input_ids, target_ids, mask = dataset[0]

        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(target_ids.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
