import gzip
import logging
import os

import numpy as np
import requests
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

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


def download_enwik8(data_dir: str = "data") -> str:
    """
    Download and prepare enwik8 dataset using Hugging Face datasets.

    @param data_dir: The directory to use for caching the dataset.
    @return: The path to the dataset cache.
    """
    # Set the cache directory for Hugging Face datasets
    os.environ["HF_DATASETS_CACHE"] = data_dir

    logger.info(
        "download_enwik8: Loading enwik8 dataset using Hugging Face datasets..."
    )
    dataset = load_dataset("enwik8", split="train")

    # Get the path to the cached dataset
    cache_path = dataset.cache_files[0]["filename"]

    logger.info(f"download_enwik8: Dataset loaded and cached at: {cache_path}")

    return cache_path


def load_and_preprocess_data(
    filepath: str, sequence_length: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess enwik8 data.

    @param filepath: The path to the dataset.
    @param sequence_length: The sequence length.
    @return: The train, validation, and test data.
    """
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # Split into train, validation, and test sets (90%, 5%, 5%)
    n = len(data)
    train_data = data[: int(0.9 * n)]
    val_data = data[int(0.9 * n) : int(0.95 * n)]
    test_data = data[int(0.95 * n) :]

    return train_data, val_data, test_data


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
