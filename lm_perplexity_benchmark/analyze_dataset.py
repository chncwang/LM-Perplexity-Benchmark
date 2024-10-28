from collections import defaultdict

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_perplexity_benchmark.utils import (
    download_wikitext103,
    setup_logger,
    tokenize_wikitext_datasets,
)

logger = setup_logger("wikitext_analysis", "wikitext_analysis.log")


def analyze_token_size_distribution(dataset: Dataset) -> dict:
    """
    Analyze the token size distribution in specified ranges.
    Returns a dictionary with percentage of samples in each range.

    @param dataset: The dataset to analyze.
    @return: A dictionary with percentage of samples in each range.
    """
    # Define token size ranges
    ranges = [
        (1, 4),
        (5, 9),
        (10, 16),
        (17, 32),
        (33, 64),
        (65, 128),
        (129, 256),
        (257, 512),
    ]

    # Count samples in each range
    distribution = defaultdict(int)
    total_samples = len(dataset)

    for sample in tqdm(dataset, desc="Analyzing token distribution"):
        input_ids = sample["input_ids"]
        # Count non-padding tokens
        token_count = sum(1 for token in input_ids if token != sample["input_ids"][-1])

        for start, end in ranges:
            if start <= token_count <= end:
                distribution[f"{start}-{end}"] += 1
                break

    # Convert counts to percentages
    percentages = {
        range_key: (count / total_samples) * 100
        for range_key, count in distribution.items()
    }

    return percentages


def analyze_datasets():
    logger.info("Starting Wikitext-103 dataset analysis")

    # Download datasets
    train_dataset, val_dataset, test_dataset = download_wikitext103()
    logger.info(
        f"analyze_datasets: Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Tokenize datasets
    model_name = "gpt2"  # Using GPT-2 tokenizer as an example
    tokenized_train, tokenized_val, tokenized_test = tokenize_wikitext_datasets(
        train_dataset, val_dataset, test_dataset, model_name
    )
    logger.info("analyze_datasets: Datasets tokenized successfully")

    # Analyze each dataset
    for name, dataset in [
        ("Training", tokenized_train),
        ("Validation", tokenized_val),
        ("Test", tokenized_test),
    ]:
        logger.info(f"analyze_datasets: Analyzing {name} dataset:")

        # Token size distribution
        distribution = analyze_token_size_distribution(dataset)
        logger.info(f"analyze_datasets: {name} token size distribution:")
        for range_key, percentage in distribution.items():
            logger.info(f"  {range_key}: {percentage:.2f}%")

        # Vocabulary analysis
        vocab_set = set()
        max_tokens = 0
        max_token_sample = None

        for sample in tqdm(
            dataset, desc=f"Analyzing {name} vocabulary"
        ):  # Add progress bar
            tokens = sample["input_ids"]
            token_count = sum(
                1 for token in tokens if token != tokens[-1]
            )  # Exclude padding
            vocab_set.update(tokens)

            if token_count > max_tokens:
                max_tokens = token_count
                max_token_sample = sample

        logger.info(f"analyze_datasets: {name} vocabulary size: {len(vocab_set)}")
        logger.info(f"analyze_datasets: {name} number of samples: {len(dataset)}")

        # Log sample with maximum tokens
        if max_token_sample:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            decoded_text = tokenizer.decode(max_token_sample["input_ids"])
            logger.info(f"analyze_datasets: {name} maximum token size: {max_tokens}")
            logger.info(
                f"analyze_datasets: {name} sample with maximum tokens:\n{decoded_text}"
            )


if __name__ == "__main__":
    analyze_datasets()
