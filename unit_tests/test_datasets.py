import logging
import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from lm_perplexity_benchmark.datasets import RnnDataset
from lm_perplexity_benchmark.utils import download_and_tokenize_wikitext103

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


class TestRnnDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across test methods."""
        # Initialize a simple tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        cls.tokenized_dataset = download_and_tokenize_wikitext103("gpt2", 512)[0]

    def test_initialization(self):
        """Test if the dataset initializes properly"""
        max_length = 512
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            max_length,
            use_cache=True,
        )

        self.assertEqual(dataset.max_length, max_length)
        self.assertEqual(dataset.pad_token_id, self.tokenizer.pad_token_id)

    def test_getitem_dimensions(self):
        """Test if __getitem__ returns tensors of correct dimensions"""
        max_length = 10
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            max_length,
            use_cache=True,
        )

        input_ids, target_ids, mask = dataset[0]
        logger.debug(
            f"test_getitem_dimensions: input_ids: {input_ids} shape: {input_ids.shape}"
        )
        logger.debug(
            f"test_getitem_dimensions: target_ids: {target_ids} shape: {target_ids.shape}"
        )
        logger.debug(f"test_getitem_dimensions: mask: {mask} shape: {mask.shape}")

        # Check dimensions
        self.assertEqual(input_ids.shape, torch.Size([max_length]))
        self.assertEqual(target_ids.shape, torch.Size([max_length]))
        self.assertEqual(mask.shape, torch.Size([max_length]))

    def test_sequence_truncation(self):
        """Test if long sequences are properly truncated"""
        max_length = 5
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            max_length,
            use_cache=True,
        )

        # Get a long sequence
        input_ids, target_ids, mask = dataset[1]

        # Check if sequences are truncated
        self.assertEqual(len(input_ids), max_length)
        self.assertEqual(len(target_ids), max_length)
        self.assertEqual(len(mask), max_length)

    def test_sequence_padding(self):
        """Test if short sequences are properly padded"""
        max_length = 10
        # Create a Dataset instead of a list
        tiny_dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        self.tokenizer.eos_token_id,
                    ],
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        self.tokenizer.eos_token_id,
                    ],
                    [1, 2, 3, 4] + [self.tokenizer.eos_token_id] * 6,
                ]
            }
        )

        dataset = RnnDataset(
            tiny_dataset,
            "tiny_test_dataset",
            self.tokenizer,
            max_length,
            use_cache=False,  # Disable cache
        )

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
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            max_length,
            use_cache=True,
        )

        input_ids, target_ids, _ = dataset[0]
        logger.debug(
            f"test_input_target_relationship: input_ids: {input_ids} shape: {input_ids.shape}"
        )
        logger.debug(
            f"test_input_target_relationship: target_ids: {target_ids} shape: {target_ids.shape}"
        )

        # Check if input is original[:-1] and target is original[1:]
        # But only for the non-padded part
        valid_length = min(len(input_ids) - 1, max_length - 1)
        logger.debug(f"test_input_target_relationship: valid_length: {valid_length}")
        logger.debug(
            f"test_input_target_relationship: input_ids[:valid_length]: {input_ids[:valid_length]} shape: {input_ids[:valid_length].shape}"
        )
        self.assertTrue(
            torch.equal(
                target_ids[:valid_length],
                torch.tensor(input_ids[1 : valid_length + 1], dtype=torch.long),
            )
        )

    def test_mask_correctness(self):
        """Test if masks correctly identify real tokens vs padding"""
        max_length = 10
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            max_length,
            use_cache=True,
        )

        # Test with a short sequence that will need padding
        input_ids, _, mask = dataset[2]

        # Count non-padding tokens in input
        real_tokens = torch.sum(input_ids != dataset.pad_token_id) + 1
        # Count True values in mask
        mask_sum = torch.sum(mask)

        # The numbers should match
        self.assertEqual(real_tokens, mask_sum)

    def test_dtype_correctness(self):
        """Test if returned tensors have correct dtypes"""
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            512,
            use_cache=True,
        )
        input_ids, target_ids, mask = dataset[0]

        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(target_ids.dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)

    def test_same_shape(self):
        """Test if input, target and mask have the same shape across the dataset and if the shapes are consistent across samples"""
        dataset = RnnDataset(
            self.tokenized_dataset,
            "empty_filtered_wikitext103_train",
            self.tokenizer,
            512,
            use_cache=True,
        )
        first_input_ids, first_target_ids, first_mask = dataset[0]
        expected_shape = first_input_ids.shape

        for i in range(min(len(dataset), 1000)):
            input_ids, target_ids, mask = dataset[i]
            logger.debug(
                f"test_same_shape: input_ids: {input_ids} shape: {input_ids.shape}"
            )
            logger.debug(
                f"test_same_shape: target_ids: {target_ids} shape: {target_ids.shape}"
            )
            logger.debug(f"test_same_shape: mask: {mask} shape: {mask.shape}")
            self.assertEqual(len(input_ids), len(target_ids))
            self.assertEqual(len(input_ids), len(mask))
            self.assertEqual(input_ids.shape, expected_shape)
            self.assertEqual(target_ids.shape, expected_shape)
            self.assertEqual(mask.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
