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
    compute_bpc,
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

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
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
                f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, BPC: {compute_bpc(loss.item()):.4f}"
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


def main():
    """
    Main function to train the model.
    """
    # Configuration
    config = {
        "data_dir": "data",
        "log_dir": "logs",
        "model_dir": "models",
        "sequence_length": 256,
        "batch_size": 128,
        "embedding_size": 400,
        "hidden_size": 1024,
        "num_layers": 2,
        "dropout": 0.5,
        "lr": 0.001,
        "clip_value": 0.25,
        "num_epochs": 40,
        "input_size": 256,  # Number of unique characters
    }

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    # Setup logger
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        "enwik8_training", os.path.join(config["log_dir"], f"training_{timestamp}.log")
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
        train_dataset, val_dataset, test_dataset, "gpt2"
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Log some samples from each dataset
    logger.info("Sample from training dataset:")
    logger.info(
        f"Text: {tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"Input IDs: {train_dataset[0]['input_ids']}")
    logger.info(f"Attention mask: {train_dataset[0]['attention_mask']}")

    logger.info("\nSample from validation dataset:")
    logger.info(
        f"Text: {tokenizer.decode(val_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"Input IDs: {val_dataset[0]['input_ids']}")
    logger.info(f"Attention mask: {val_dataset[0]['attention_mask']}")

    logger.info("\nSample from test dataset:")
    logger.info(
        f"Text: {tokenizer.decode(test_dataset[0]['input_ids'], skip_special_tokens=True)}"
    )
    logger.info(f"Input IDs: {test_dataset[0]['input_ids']}")
    logger.info(f"Attention mask: {test_dataset[0]['attention_mask']}")

    exit()

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config["batch_size"],
    )

    # Initialize model
    model = LSTMModel(
        config["input_size"],
        config["embedding_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}')
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["clip_value"],
            logger,
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        train_bpc = compute_bpc(train_loss)
        val_bpc = compute_bpc(val_loss)

        epoch_time = time.time() - start_time

        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"Train Loss: {train_loss:.4f}, Train BPC: {train_bpc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val BPC: {val_bpc:.4f}")
        logger.info(f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                os.path.join(config["model_dir"], "best_model.pt"),
            )
            logger.info("Saved best model")

    # Final evaluation on test set
    model.load_state_dict(
        torch.load(os.path.join(config["model_dir"], "best_model.pt"))[
            "model_state_dict"
        ]
    )
    test_loss = evaluate(model, test_loader, criterion, device)
    test_bpc = compute_bpc(test_loss)
    logger.info(f"Test Loss: {test_loss:.4f}, Test BPC: {test_bpc:.4f}")


if __name__ == "__main__":
    main()
