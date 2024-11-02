import gzip
import logging
import os

import numpy as np
import requests
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from lm_perplexity_benchmark.datasets import RnnDataset

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

    # Set urllib3.connectionpool logging level to INFO
    urllib3_logger = logging.getLogger("urllib3.connectionpool")
    urllib3_logger.setLevel(logging.INFO)

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


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    dataset_name: str,
    batch_size: int,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for training, validation, and testing using Hugging Face Datasets.

    @param train_dataset: The training dataset.
    @param val_dataset: The validation dataset.
    @param test_dataset: The test dataset.
    @param dataset_name: The name of the dataset.
    @param batch_size: The batch size.
    @param tokenizer: The tokenizer.
    @param max_length: The maximum sequence length.
    @return: The train, validation, and test DataLoader objects.
    """
    # Wrap the Hugging Face datasets with RnnDataset
    train_dataset = RnnDataset(
        train_dataset,
        f"empty_filtered_{dataset_name}_train",
        tokenizer,
        max_length=max_length,
    )
    val_dataset = RnnDataset(
        val_dataset,
        f"empty_filtered_{dataset_name}_val",
        tokenizer,
        max_length=max_length,
    )
    test_dataset = RnnDataset(
        test_dataset,
        f"empty_filtered_{dataset_name}_test",
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
