import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch._dynamo import config as dynamo_config
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_perplexity_benchmark.model import CustomLSTM, LSTMModel
from lm_perplexity_benchmark.utils import (
    create_dataloaders,
    download_and_tokenize_wikitext103,
    setup_logger,
)

# Configure PyTorch tensor logging globally
torch.set_printoptions(
    precision=3, threshold=100, edgeitems=1, linewidth=80, profile=None, sci_mode=False
)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_value: float,
    logger: logging.Logger,
    tokenizer: AutoTokenizer,
    scaler: GradScaler,
    epoch: int,
    checkpoint_dir: str,
    checkpoint_interval: int,
    start_batch_idx: int = 0,
) -> tuple[float, int]:
    """
    Train one epoch of the model.
    """
    model.train()
    total_loss = 0
    hidden = None
    smoothed_loss = 0

    # Add timing variables
    start_time = time.time()
    total_tokens = 0
    tokens_per_second = 0

    last_checkpoint_time = time.time()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    for batch_idx, (input_ids, target_ids, mask) in enumerate(
        tqdm(train_loader, desc="Training", initial=start_batch_idx)
    ):
        if batch_idx < start_batch_idx:
            continue

        # Count tokens in this batch (sum of True values in mask)
        batch_tokens = mask.sum().item()
        total_tokens += batch_tokens

        if batch_idx % 100 == 0:
            logger.debug(
                f"train_epoch: Batch {batch_idx}: input shape: {input_ids.shape}, target shape: {target_ids.shape}, mask shape: {mask.shape}"
            )

        # Move everything to device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Detach hidden states
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        # Wrap forward and loss computation in autocast
        with autocast(enabled=device.type == "cuda"):
            output = model(input_ids)

            # Calculate loss only on non-padded positions
            output_for_loss = output.view(-1, output.size(-1))
            target_ids_for_loss = target_ids.view(-1)
            mask_for_loss = mask.view(-1)

            output_for_loss = output_for_loss[mask_for_loss]
            target_ids_for_loss = target_ids_for_loss[mask_for_loss]

            loss = criterion(output_for_loss, target_ids_for_loss)

        # Replace manual backward with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), clip_value)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Update smoothed loss
        smoothed_loss = 0.999 * smoothed_loss + 0.001 * loss.item()

        if batch_idx % 100 == 0:
            # Calculate tokens per second
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:  # Avoid division by zero
                tokens_per_second = total_tokens / elapsed_time

            logger.info(
                f"train_epoch: Batch {batch_idx}/{len(train_loader)}, "
                f"Smoothed Loss: {smoothed_loss:.4f}, "
                f"Tokens/sec: {tokens_per_second:.2f}"
            )
            logger.debug(f"train_epoch: mask[0]: {mask[0]}")
            # Decode the input sequence
            # Apply mask to get only valid tokens before decoding
            valid_tokens = input_ids[0][mask[0]]
            decoded_sequence = tokenizer.decode(valid_tokens, skip_special_tokens=False)
            logger.info(f"train_epoch: Decoded Input Sequence: {decoded_sequence}")

            # Log the target sequence at debug level
            valid_target_tokens = target_ids[0][mask[0]]
            decoded_target = tokenizer.decode(
                valid_target_tokens, skip_special_tokens=False
            )
            logger.debug(f"train_epoch: Target Sequence: {decoded_target}")

            # Log the model's predicted tokens
            predicted_ids = torch.argmax(output[0], dim=-1)
            valid_predicted_ids = predicted_ids[mask[0]]
            decoded_predictions = tokenizer.decode(
                valid_predicted_ids, skip_special_tokens=False
            )
            logger.info(f"train_epoch: Predicted Tokens: {decoded_predictions}")

        # Add checkpoint saving
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval:
            train_loader_state = {
                "batch_idx": batch_idx,
                "dataset_state": getattr(
                    train_loader.dataset, "__getstate__", lambda: {}
                )(),
            }

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                batch_idx,
                train_loader_state,
                total_loss / (batch_idx + 1),
                checkpoint_path,
                logger,
            )
            last_checkpoint_time = current_time

    return total_loss / len(train_loader), batch_idx


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    hidden = None

    with torch.no_grad():
        for input_ids, target_ids, mask in eval_loader:
            # Move everything to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            mask = mask.to(device)

            # Forward pass
            output = model(input_ids)

            # Calculate loss only on non-padded positions
            output = output.view(-1, output.size(-1))
            target_ids = target_ids.view(-1)
            mask = mask.view(-1)

            # Apply mask to exclude padding tokens from loss calculation
            output = output[mask]
            target_ids = target_ids[mask]

            loss = criterion(output, target_ids)
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
    parser.add_argument(
        "--lstm_class",
        type=str,
        default="nn.LSTM",
        help="LSTM class to use (nn.LSTM or CustomLSTM)",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=0.25, help="Gradient clipping value"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (None for unlimited)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (number of epochs without improvement). Set to None to disable early stopping",
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
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--disable_half_precision",
        action="store_true",
        help="Disable automatic mixed precision training",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="Disable PyTorch 2.0 compilation (not recommended)",
    )
    parser.add_argument(
        "--hippo_dim",
        type=int,
        default=32,
        help="Dimension of the hippo state",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=3600,  # 1 hour in seconds
        help="Time interval between checkpoints in seconds",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    return parser.parse_args()


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    batch_idx: int,
    train_loader_state: dict,
    loss: float,
    checkpoint_path: str,
    logger: logging.Logger,
):
    """Save model checkpoint with training state."""
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loader_state": train_loader_state,
        "random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint at epoch {epoch}, batch {batch_idx}")


def main():
    args = parse_args()

    # Setup logger first
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        "wikitext103_training",
        os.path.join(
            "logs", f"training_{timestamp}.log"
        ),  # Using "logs" directly since config isn't created yet
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    # Now we can use logger
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
        "weight_decay": args.weight_decay,
        "disable_half_precision": args.disable_half_precision,
        "lstm_class": args.lstm_class,
        "hippo_dim": args.hippo_dim,
    }
    logger.info(f"main: Hyperparameters: {hyperparameters}")

    # Configuration
    config = {
        "data_dir": "data",
        "log_dir": "logs",
        "model_dir": "models",
    }
    logger.info(f"main: Config: {config}")

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

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
    train_dataset, val_dataset, test_dataset = download_and_tokenize_wikitext103(
        "gpt2", hyperparameters["max_length"]
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Calculate vocabulary size
    vocab_size = len(tokenizer)
    logger.info(f"main: Vocabulary size: {vocab_size}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        "wikitext103",
        hyperparameters["batch_size"],
        tokenizer,
        hyperparameters["max_length"],
    )

    # Initialize model
    model = LSTMModel(
        vocab_size,
        hyperparameters["embedding_size"],
        hyperparameters["hidden_size"],
        hyperparameters["num_layers"],
        hyperparameters["dropout"],
        hippo_dim=hyperparameters["hippo_dim"],
        lstm_class=nn.LSTM if args.lstm_class == "nn.LSTM" else CustomLSTM,
    ).to(device)

    # Only compile if using CUDA and compilation is not disabled
    if not args.disable_compile and device.type == "cuda":
        logger.info("main: Compiling model with mode: reduce-overhead")
        # Set higher threshold for better compilation speed
        dynamo_config.cache_size_limit = 512
        model = torch.compile(model, mode="reduce-overhead")
    else:
        logger.info(
            "main: Model compilation skipped (either disabled or not using CUDA)"
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters["lr"],
        weight_decay=hyperparameters["weight_decay"],
    )

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Initialize the GradScaler
    scaler = GradScaler(
        enabled=device.type == "cuda" and not args.disable_half_precision
    )

    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0

    start_epoch = 0
    start_batch_idx = 0

    # Load checkpoint if resuming
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint["batch_idx"] + 1

        if "train_loader_state" in checkpoint:
            train_dataset_state = checkpoint["train_loader_state"].get(
                "dataset_state", {}
            )
            if hasattr(train_dataset, "__setstate__"):
                train_dataset.__setstate__(train_dataset_state)

        # Restore random states
        torch.set_rng_state(checkpoint["random_state"])
        if checkpoint["cuda_random_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_random_state"])

    # Update training loop
    epoch = start_epoch
    while True:
        if args.num_epochs is not None and epoch >= args.num_epochs:
            break

        logger.info(f"main: Epoch {epoch+1}")
        start_time = time.time()

        train_loss, last_batch_idx = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            hyperparameters["clip_value"],
            logger,
            tokenizer,
            scaler,
            epoch,
            args.checkpoint_dir,
            args.checkpoint_interval,
            start_batch_idx if epoch == start_epoch else 0,
        )

        start_batch_idx = 0  # Reset after first resumed epoch

        val_loss = evaluate(model, val_loader, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        # Step the scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        logger.info(f"main: Epoch {epoch+1}:")
        logger.info(f"main: Train Loss: {train_loss:.4f}")
        logger.info(f"main: Val Loss: {val_loss:.4f}")
        logger.info(f"main: Test Loss: {test_loss:.4f}")
        logger.info(f"main: Time: {epoch_time:.2f}s")

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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
        else:
            patience_counter += 1
            if args.patience is not None and patience_counter >= args.patience:
                logger.info(f"main: Early stopping triggered after {epoch+1} epochs")
                break

        epoch += 1


if __name__ == "__main__":
    main()
