import logging
import os
import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RnnDataset(Dataset):
    """
    Dataset class for RNN-based language models that handles token sequences properly.
    Filters out empty sequences during initialization.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        use_cache: bool = False,
    ):
        """
        Initialize the RNN dataset.
        @param dataset: An enumeratable object where each element is a dictionary containing an 'input_ids' field
        @param dataset_name: Name of the dataset
        @param tokenizer: The tokenizer used for getting special tokens
        @param max_length: Maximum sequence length
        @param use_cache: Whether to use cached indices or not
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cache = use_cache
        # Ensure we have the necessary special tokens
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

        # Get the start token ID, fallback to BOS token if not available
        self.start_token_id = tokenizer.cls_token_id
        if self.start_token_id is None:
            self.start_token_id = tokenizer.bos_token_id
        if self.start_token_id is None:
            logger.warning(
                "No start token found in tokenizer, using EOS token as start token"
            )
            self.start_token_id = tokenizer.eos_token_id

        # Ensure the cache directory exists
        cache_dir = "./cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Set cache file name based on the dataset's path
        cache_file: str = os.path.join(cache_dir, f"{dataset_name}_cache.pkl")
        logger.info(f"RnnDataset.__init__: cache_file: {cache_file}")

        # Check if cache file exists and use_cache is True
        if self.use_cache and os.path.exists(cache_file):
            logger.info("RnnDataset.__init__: Loading cached indices...")
            with open(cache_file, "rb") as f:
                non_empty_indices = pickle.load(f)
        else:
            logger.info("RnnDataset.__init__: Filtering out empty sequences...")
            non_empty_indices = [
                i
                for i, item in enumerate(tqdm(dataset, desc="Filtering sequences"))
                if item["input_ids"][0] != self.tokenizer.eos_token_id
            ]
            filtered_percentage = (
                (len(dataset) - len(non_empty_indices)) / len(dataset) * 100
            )
            logger.info(
                f"RnnDataset.__init__: Filtered out {len(dataset) - len(non_empty_indices)} empty sequences "
                f"({filtered_percentage:.2f}% of total, {len(non_empty_indices)} sequences remaining)"
            )
            # Save indices to cache file if use_cache is True
            if self.use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(non_empty_indices, f)

        if len(non_empty_indices) < len(dataset):
            # Create a new filtered dataset
            self.dataset = dataset.select(non_empty_indices)
        else:
            self.dataset = dataset

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

        # Raise an error if the first token is eos_token
        if token_ids[0] == self.tokenizer.eos_token_id:
            raise ValueError(
                f"First token is eos_token_id ({self.tokenizer.eos_token_id})"
            )

        # Ensure we don't exceed max_length (accounting for start token)
        if len(token_ids) >= self.max_length:
            token_ids = token_ids[: self.max_length]

        # Create input sequence starting with start token
        input_ids = [self.start_token_id] + token_ids[:-1]
        # Create target sequence starting with the first actual token
        target_ids = token_ids

        # Find the position of the first EOS token after the actual tokens
        # Skip the very first token in case it's an EOS token at the start
        eos_positions = [
            i
            for i, token in enumerate(input_ids[1:], start=1)
            if token == self.tokenizer.eos_token_id
        ]
        seq_length = len(input_ids) if not eos_positions else eos_positions[0]
        if seq_length > self.max_length:
            logger.error(f"RnnDataset.__getitem__: input_ids: {input_ids}")
            logger.error(f"RnnDataset.__getitem__: eos_positions: {eos_positions}")
            raise ValueError(
                f"Sequence length ({seq_length}) exceeds max_length ({self.max_length})"
            )

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:seq_length] = True

        # Raise an error if the three tensors are not of the same length, namely self.max_length
        if (
            len(input_ids) != len(target_ids)
            or len(input_ids) != len(mask)
            or len(mask) != self.max_length
        ):
            logger.error(
                f"RnnDataset.__getitem__: len(input_ids): {len(input_ids)} len(target_ids): {len(target_ids)} len(mask): {len(mask)}"
            )
            logger.error(f"RnnDataset.__getitem__: input_ids: {input_ids}")
            logger.error(f"RnnDataset.__getitem__: target_ids: {target_ids}")
            logger.error(f"RnnDataset.__getitem__: mask: {mask}")
            raise ValueError(
                "Input, target, and mask tensors are not of the same length"
            )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            mask,
        )
