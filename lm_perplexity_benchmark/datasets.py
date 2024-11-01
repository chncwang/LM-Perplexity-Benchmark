import logging

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RnnDataset(Dataset):
    """
    Dataset class for RNN-based language models that handles token sequences properly.
    """

    def __init__(
        self, dataset: HFDataset, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ):
        """
        Initialize the RNN dataset.

        @param dataset: HuggingFace dataset containing tokenized text
        @param tokenizer: The tokenizer used for getting special tokens
        @param max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure we have the necessary special tokens
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        @param idx: Index of the item
        @return: Tuple of (input_ids, target_ids, mask)
        """
        # Get the tokenized sequence
        item = self.dataset[idx]
        token_ids = item["input_ids"]

        # Ensure we don't exceed max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        # Create input sequence (all tokens except last)
        input_ids = token_ids[:-1]
        # Create target sequence (all tokens except first)
        target_ids = token_ids[1:]

        # Calculate actual sequence length
        seq_length = len(input_ids)

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.ones(seq_length, dtype=torch.bool)

        # Pad sequences if necessary
        if seq_length < self.max_length - 1:  # -1 because we removed one token
            padding_length = self.max_length - 1 - seq_length
            # Pad input_ids
            input_ids.extend([self.pad_token_id] * padding_length)
            # Pad target_ids
            target_ids.extend([self.pad_token_id] * padding_length)
            # Update mask for padding
            mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.bool)])

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            mask,
        )
