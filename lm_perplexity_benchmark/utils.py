import gzip
import logging
import os

import numpy as np
import requests
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that writes to both file and console.

    @param name: The name of the logger.
    @param log_file: The file to write the log to.
    @param level: The level of the logger.
    @return: The logger.
    """
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            file_handler,
            stream_handler,
        ],
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def download_wikitext103() -> tuple[Dataset, Dataset, Dataset]:
    """
    Download and prepare wikitext-103 dataset using Hugging Face datasets.

    @return: A tuple of (train, validation, test) datasets.
    """
    logger.info(
        "download_wikitext103: Loading wikitext-103 dataset using Hugging Face datasets..."
    )
    # Load all splits
    train_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    test_dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

    logger.info("download_wikitext103: Datasets loaded successfully")

    return train_dataset, val_dataset, test_dataset


def tokenize_wikitext_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    model_name: str,
    max_length: int = 512,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize the Wikitext datasets using a tokenizer from Hugging Face.

    @param train_dataset: The training dataset
    @param val_dataset: The validation dataset
    @param test_dataset: The test dataset
    @param model_name: The model name for selecting the tokenizer
    @param max_length: The maximum length for token sequences (default is 512)
    @return: Tuple of tokenized (train, validation, test) datasets
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        cache_file_name=os.path.join(cache_dir, "tokenized_train.arrow"),
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        cache_file_name=os.path.join(cache_dir, "tokenized_val.arrow"),
    )
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        cache_file_name=os.path.join(cache_dir, "tokenized_test.arrow"),
    )

    return tokenized_train, tokenized_val, tokenized_test


class TextDataset(Dataset):
    """
    Dataset class for character-level text data.
    """

    def __init__(self, data: np.ndarray, sequence_length: int) -> None:
        """
        Initialize the dataset.

        @param data: The data.
        @param sequence_length: The sequence length.
        """
        self.data = torch.from_numpy(data).long()
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        @return: The length of the dataset.
        """
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        @param idx: The index.
        @return: A tuple of tensors.
        """
        x = self.data[idx : idx + self.sequence_length]
        y = self.data[idx + 1 : idx + self.sequence_length + 1]
        return x, y


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    sequence_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for training, validation, and testing.

    @param train_data: The training data.
    @param val_data: The validation data.
    @param test_data: The test data.
    @param sequence_length: The sequence length.
    @param batch_size: The batch size.
    @return: The train, validation, and test DataLoader objects.
    """
    train_dataset = TextDataset(train_data, sequence_length)
    val_dataset = TextDataset(val_data, sequence_length)
    test_dataset = TextDataset(test_data, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def compute_bpc(loss: float) -> float:
    """
    Convert loss to bits per character (BPC).

    @param loss: The loss.
    @return: The BPC.
    """
    return loss / np.log(2)
