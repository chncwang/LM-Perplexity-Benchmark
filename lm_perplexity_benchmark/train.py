import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_perplexity_benchmark.model import LSTMModel
from lm_perplexity_benchmark.utils import (
    create_dataloaders,
    download_wikitext103,
    setup_logger,
    tokenize_wikitext_datasets,
)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_value: float,
    logger: logging.Logger,
) -> float:
    """
    Train one epoch of the model.

    @param model: The model to train.
    @param train_loader: The training data loader.
    @param criterion: The loss criterion.
    @param optimizer: The optimizer.
    @param device: The device to train on.
    @param clip_value: The gradient clipping value.
    @param logger: The logger.
    @return: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    hidden = None

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        logger.debug(f"train_epoch: Batch {batch_idx}: {batch}")
        data, target = batch[:2]
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Detach hidden states
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))

        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logger.info(
                f"train_epoch: Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set.

    @param model: The model to evaluate.
    @param eval_loader: The validation data loader.
    @param criterion: The loss criterion.
    @param device: The device to evaluate on.
    @return: The average loss for the validation set.
    """
    model.eval()
    total_loss = 0
    hidden = None

    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)

            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()

            # Detach hidden states
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden)

    return total_loss / len(eval_loader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM model on Wikitext-103 dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="Size of the embedding layer"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Number of hidden units in LSTM"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=0.25, help="Gradient clipping value"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    args = parse_args()

    hyperparameters = {
        "batch_size": args.batch_size,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "clip_value": args.clip_value,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
    }

    # Configuration
    config = {
        "data_dir": "data",
        "log_dir": "logs",
        "model_dir": "models",
    }

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    # Setup logger
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        "wikitext103_training",
        os.path.join(config["log_dir"], f"training_{timestamp}.log"),
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    # Save configuration
    with open(os.path.join(config["log_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = download_wikitext103()
    train_dataset, val_dataset, test_dataset = tokenize_wikitext_datasets(
        train_dataset, val_dataset, test_dataset, "gpt2", hyperparameters["max_length"]
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Calculate vocabulary size
    vocab_size = len(tokenizer)
    logger.info(f"main: Vocabulary size: {vocab_size}")

    # Log some samples from each dataset
    logger.info("main: Sample from training dataset:")
    logger.info(
        f"Text: {tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"main: Input IDs: {train_dataset[0]['input_ids']}")
    logger.info(f"main: Attention mask: {train_dataset[0]['attention_mask']}")

    logger.info("main: Sample from validation dataset:")
    logger.info(
        f"Text: {tokenizer.decode(val_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"main: Input IDs: {val_dataset[0]['input_ids']}")
    logger.info(f"main: Attention mask: {val_dataset[0]['attention_mask']}")

    logger.info("main: Sample from test dataset:")
    logger.info(
        f"Text: {tokenizer.decode(test_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"main: Input IDs: {test_dataset[0]['input_ids']}")
    logger.info(f"main: Attention mask: {test_dataset[0]['attention_mask']}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        hyperparameters["batch_size"],
    )

    # Initialize model
    model = LSTMModel(
        vocab_size,
        hyperparameters["embedding_size"],
        hyperparameters["hidden_size"],
        hyperparameters["num_layers"],
        hyperparameters["dropout"],
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(hyperparameters["num_epochs"]):
        logger.info(f"main: Epoch {epoch+1}/{hyperparameters['num_epochs']}")
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            hyperparameters["clip_value"],
            logger,
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        logger.info(f"main: Epoch {epoch+1}:")
        logger.info(f"main: Train Loss: {train_loss:.4f}")
        logger.info(f"main: Val Loss: {val_loss:.4f}")
        logger.info(f"main: Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "hyperparameters": hyperparameters,
                },
                os.path.join(config["model_dir"], "best_model.pt"),
            )
            logger.info("main: Saved best model")

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(config["model_dir"], "best_model.pt"))[
            "model_state_dict"
        ]
    )
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info(f"main: Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
