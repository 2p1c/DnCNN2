"""
Training Pipeline for Ultrasonic Signal Denoising CAE

Includes:
- Training loop with PSNR monitoring
- Pre-training visualization (noisy vs clean samples)
- Post-training results visualization (noisy vs clean vs denoised)
- Model checkpoint saving with best validation PSNR

Usage:
    uv run python train.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules like model, data_utils can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import inspect
import secrets
from pathlib import Path
from typing import Tuple, Optional, Dict
from tqdm import tqdm

from data import create_dataloaders
from model.model import (
    DeepCAE,
    DeeperCAE,
    LightweightCAE,
    UnsupervisedDeepCAE,
    count_parameters,
)
from scripts.analysis.acoustic_validation import run_acoustic_validation
from scripts.train.visualization import plot_results


RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def _image_path(filename: str) -> str:
    """Build a default image output path under results/images."""
    return str(IMAGES_DIR / filename)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for an output file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def calculate_psnr(
    clean: torch.Tensor, denoised: torch.Tensor, max_val: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX² / MSE)

    Higher PSNR indicates better reconstruction quality.
    Typical values: 20-40 dB for good denoising.

    Args:
        clean: Ground truth signal tensor
        denoised: Denoised/reconstructed signal tensor
        max_val: Maximum possible pixel/signal value (1.0 for normalized)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((clean - denoised) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def calculate_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Calculate Signal-to-Noise Ratio.

    SNR = 10 * log10(P_signal / P_noise)

    Args:
        signal: Clean signal tensor
        noise: Noise tensor (noisy - clean)

    Returns:
        SNR value in dB
    """
    signal_power = torch.mean(signal**2).item()
    noise_power = torch.mean(noise**2).item()
    if noise_power < 1e-10:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def plot_pre_training_samples(
    dataloader: torch.utils.data.DataLoader,
    save_path: str = _image_path("fig_pre_train_samples.png"),
    num_samples: int = 3,
) -> None:
    """
    Plot noisy vs clean samples before training starts.

    This visualization helps verify that:
    1. Data generation is working correctly
    2. Noise levels are appropriate
    3. Signal features are visible

    Args:
        dataloader: DataLoader to sample from
        save_path: Path to save the figure
        num_samples: Number of sample pairs to plot
    """
    # Get one batch of data
    noisy, clean = next(iter(dataloader))

    # Create figure with 2 columns: Noisy | Clean
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    fig.suptitle(
        "Pre-Training Samples: Noisy Input vs Clean Ground Truth",
        fontsize=14,
        fontweight="bold",
    )

    # Time axis (in microseconds for display)
    time_us = np.linspace(0, 160, 1000)  # 160 μs duration

    for i in range(num_samples):
        noisy_signal = noisy[i, 0].numpy()
        clean_signal = clean[i, 0].numpy()

        # Calculate input SNR
        noise = noisy_signal - clean_signal
        snr = calculate_snr(torch.from_numpy(clean_signal), torch.from_numpy(noise))

        # Plot noisy signal
        axes[i, 0].plot(time_us, noisy_signal, "b-", linewidth=0.7, alpha=0.8)
        axes[i, 0].set_title(f"Sample {i + 1}: Noisy Input (SNR: {snr:.1f} dB)")
        axes[i, 0].set_xlabel("Time (μs)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_ylim(-2.5, 2.5)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

        # Plot clean signal
        axes[i, 1].plot(time_us, clean_signal, "g-", linewidth=0.8)
        axes[i, 1].set_title(f"Sample {i + 1}: Clean Ground Truth")
        axes[i, 1].set_xlabel("Time (μs)")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].set_ylim(-2.5, 2.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved pre-training samples to {save_path}")


def plot_training_curves(
    history: Dict[str, list], save_path: str = _image_path("fig_training_curves.png")
) -> None:
    """
    Plot training and validation loss/PSNR curves.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_psnr', 'val_psnr'
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # PSNR curve
    axes[1].plot(epochs, history["train_psnr"], "b-", label="Train PSNR", linewidth=1.5)
    axes[1].plot(epochs, history["val_psnr"], "r-", label="Val PSNR", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved training curves to {save_path}")


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    unsupervised: bool = False,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        device: Device to train on
        unsupervised: If True, use noisy input as reconstruction target

    Returns:
        Tuple of (average_loss, average_psnr)
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        target = noisy if unsupervised else clean

        # Forward pass
        denoised = model(noisy)
        loss = criterion(denoised, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_psnr += calculate_psnr(clean, denoised)
        num_batches += 1

    return total_loss / num_batches, total_psnr / num_batches


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    unsupervised: bool = False,
) -> Tuple[float, float]:
    """
    Validate the model on validation set.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        unsupervised: If True, use noisy input as reconstruction target

    Returns:
        Tuple of (average_loss, average_psnr)
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            target = noisy if unsupervised else clean

            denoised = model(noisy)
            loss = criterion(denoised, target)

            total_loss += loss.item()
            total_psnr += calculate_psnr(clean, denoised)
            num_batches += 1

    return total_loss / num_batches, total_psnr / num_batches


def train(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_train: int = 5000,
    num_val: int = 1000,
    save_best: bool = True,
    checkpoint_dir: str = str(CHECKPOINTS_DIR),
    use_deeper_model: bool = False,
    model_type: str = "lightweight",  # 'lightweight', 'deeper', 'deep', 'unsupervised_deep'
    seed: Optional[int] = None,
    data_mode: str = "file",
    data_path: Optional[str] = "data",
    early_stopping_patience: int = 50,  # More patient: wait 50 epochs without improvement
    min_epochs: int = 30,  # Minimum epochs before early stopping can trigger
    dropout_rate: float = 0.1,  # Lower dropout (0.1) - 0.4 was too aggressive
    augment: bool = False,  # Enable data augmentation for training set
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Main training function.

    Implements the complete training pipeline:
    1. Setup device and dataloaders
    2. Initialize model, loss, optimizer
    3. Generate pre-training visualization
    4. Training loop with validation
    5. Save best model checkpoint
    6. Generate post-training visualization

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        num_train: Number of synthetic training samples (only for synthetic mode)
        num_val: Number of synthetic validation samples (only for synthetic mode)
        save_best: Whether to save best model based on val PSNR
        checkpoint_dir: Directory to save model checkpoints
        use_deeper_model: Deprecated, use model_type instead
        model_type: Model architecture - 'lightweight' (3层), 'deeper' (4层), 'deep' (5层), 'unsupervised_deep' (无监督5层)
        seed: Random seed for reproducibility. If None or negative, generate a random seed.
        data_mode: 'synthetic' or 'file'
        data_path: Path to data directory (for file mode, output of transformer.py)
        early_stopping_patience: Stop training if validation PSNR doesn't improve for N epochs
        dropout_rate: Dropout probability for model regularization (higher = more regularization)

    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set seeds for reproducibility (or randomize when seed is None/negative)
    if seed is None or seed < 0:
        resolved_seed = int(secrets.randbits(32))
        print(f"[INFO] Random seed generated: {resolved_seed}")
    else:
        resolved_seed = int(seed)
        if resolved_seed > 2**32 - 1:
            raise ValueError("seed must be <= 2**32 - 1")
        print(f"[INFO] Using fixed seed: {resolved_seed}")

    torch.manual_seed(resolved_seed)
    np.random.seed(resolved_seed)

    # ============================================================
    # Device Selection
    # ============================================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")

    # ============================================================
    # Create Dataloaders
    # ============================================================
    if data_mode == "file":
        if data_path is None:
            data_path = "data"
            print("[WARNING] data_path is None in file mode, fallback to 'data'.")
        print(f"\n[INFO] Loading experimental data from {data_path}...")
        train_loader, val_loader = create_dataloaders(
            batch_size=batch_size,
            seed=resolved_seed,
            mode="file",
            data_path=data_path,
            augment=augment,
        )
        print("[INFO] Data mode: FILE (experimental data)")
        if augment:
            print("[INFO] Data augmentation: ENABLED")
    else:
        print("\n[INFO] Creating synthetic datasets...")
        train_loader, val_loader = create_dataloaders(
            num_train=num_train,
            num_val=num_val,
            batch_size=batch_size,
            seed=resolved_seed,
            mode="synthetic",
            augment=augment,
        )
        print("[INFO] Data mode: SYNTHETIC")
        print(f"[INFO] Training samples: {num_train}, Validation samples: {num_val}")
        if augment:
            print("[INFO] Data augmentation: ENABLED")

    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ============================================================
    # Initialize Model
    # ============================================================
    # Support legacy use_deeper_model parameter
    if use_deeper_model and model_type == "lightweight":
        model_type = "deeper"

    is_unsupervised = model_type == "unsupervised_deep"

    if model_type == "deep":
        model = DeepCAE(dropout_rate=dropout_rate).to(device)
        print("\n[INFO] Using DeepCAE model (5层, 宽通道)")
    elif model_type == "unsupervised_deep":
        model = UnsupervisedDeepCAE(dropout_rate=dropout_rate).to(device)
        print("\n[INFO] Using UnsupervisedDeepCAE model (5层, 输入重建)")
    elif model_type == "deeper":
        model = DeeperCAE(dropout_rate=dropout_rate).to(device)
        print("\n[INFO] Using DeeperCAE model (4层)")
    else:
        model = LightweightCAE(dropout_rate=dropout_rate).to(device)
        print("\n[INFO] Using LightweightCAE model (3层)")

    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    print(f"[INFO] Dropout rate: {dropout_rate}")

    # ============================================================
    # Loss Function and Optimizer
    # ============================================================
    criterion = nn.MSELoss()  # Minimizing MSE maximizes PSNR
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )  # Stronger L2 regularization

    # Learning rate scheduler: Cosine Annealing with Warm Restarts
    # Better for deep networks - allows escaping local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,  # Initial restart period
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6,  # Minimum learning rate
    )

    print(f"[INFO] Optimizer: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print("[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0=25)")
    print(
        f"[INFO] Early Stopping: patience={early_stopping_patience}, min_epochs={min_epochs}"
    )
    if is_unsupervised:
        print("[INFO] Loss: MSELoss (target = noisy input, unsupervised)")
    else:
        print("[INFO] Loss: MSELoss (target = clean signal)")

    # ============================================================
    # Create Checkpoint Directory
    # ============================================================
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Pre-Training Visualization
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] Generating pre-training samples visualization...")
    print("=" * 60)
    plot_pre_training_samples(train_loader, _image_path("fig_pre_train_samples.png"))

    # ============================================================
    # Training Loop
    # ============================================================
    print("\n" + "=" * 60)
    print(f"[INFO] Starting training for {num_epochs} epochs...")
    print("=" * 60 + "\n")

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_psnr": [],
        "val_psnr": [],
    }

    best_val_psnr = -float("inf")
    best_checkpoint_name = (
        "best_unsupervised_model.pth" if is_unsupervised else "best_model.pth"
    )
    early_stopping_counter = 0  # Counter for early stopping

    # Progress bar
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:
        # Train one epoch
        train_loss, train_psnr = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            unsupervised=is_unsupervised,
        )

        # Validate
        val_loss, val_psnr = validate(
            model,
            val_loader,
            criterion,
            device,
            unsupervised=is_unsupervised,
        )

        # Update learning rate scheduler (CosineAnnealingWarmRestarts uses epoch)
        scheduler.step(epoch)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(train_psnr)
        history["val_psnr"].append(val_psnr)

        # Update progress bar
        pbar.set_postfix(
            {"train_loss": f"{train_loss:.6f}", "val_psnr": f"{val_psnr:.2f}dB"}
        )

        # Print detailed progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            gap = train_psnr - val_psnr  # Overfitting indicator
            tqdm.write(
                f"Epoch [{epoch:3d}/{num_epochs}] | "
                f"Train Loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f} dB | "
                f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | "
                f"Gap: {gap:.2f} dB | LR: {current_lr:.2e}"
            )

        # Save best model and check early stopping
        if save_best and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            early_stopping_counter = 0  # Reset counter
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_loss": val_loss,
                    "train_psnr": train_psnr,
                    "train_loss": train_loss,
                    "unsupervised": is_unsupervised,
                },
                checkpoint_path / best_checkpoint_name,
            )
            tqdm.write(f"  → Saved new best model (Val PSNR: {val_psnr:.2f} dB)")
        else:
            early_stopping_counter += 1
            # Only trigger early stopping after minimum epochs
            if (
                epoch >= min_epochs
                and early_stopping_counter >= early_stopping_patience
            ):
                tqdm.write(
                    f"\n[INFO] Early stopping triggered at epoch {epoch}! No improvement for {early_stopping_patience} epochs."
                )
                break

    # ============================================================
    # Load Best Model
    # ============================================================
    if save_best and (checkpoint_path / best_checkpoint_name).exists():
        checkpoint = torch.load(
            checkpoint_path / best_checkpoint_name,
            map_location=device,
            weights_only=False,  # Trust our own checkpoint
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"\n[INFO] Loaded best model from epoch {checkpoint['epoch']} "
            f"(Val PSNR: {checkpoint['val_psnr']:.2f} dB)"
        )

    # ============================================================
    # Post-Training Visualizations
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] Generating post-training visualizations...")
    print("=" * 60)

    # Build training config for display
    train_config = {
        "model": model_type,
        "epochs": num_epochs,
        "dropout": dropout_rate,
        "augment": augment,
        "mode": data_mode,
        "best_psnr": best_val_psnr,
    }

    results_image_name = "fig_unsupervised_results.png" if is_unsupervised else "fig_results.png"
    curves_image_name = (
        "fig_unsupervised_training_curves.png"
        if is_unsupervised
        else "fig_training_curves.png"
    )
    acoustic_image_name = (
        "fig_unsupervised_acoustic_validation.png"
        if is_unsupervised
        else "fig_acoustic_validation.png"
    )

    # Results visualization
    plot_results(
        model,
        val_loader,
        device,
        _image_path(results_image_name),
        train_config=train_config,
        unsupervised=is_unsupervised,
    )

    # Training curves
    plot_training_curves(history, _image_path(curves_image_name))

    # Acoustic feature validation (声学特征验证)
    print("\n" + "=" * 60)
    print("[INFO] Running acoustic feature validation...")
    print("=" * 60)
    run_acoustic_validation(
        model,
        val_loader,
        device,
        save_path=_image_path(acoustic_image_name),
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] Training Complete!")
    print("=" * 60)
    print(f"  → Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"  → Final training PSNR: {history['train_psnr'][-1]:.2f} dB")
    print("  → Figures saved:")
    print(f"      - {_image_path('fig_pre_train_samples.png')}")
    print(f"      - {_image_path(results_image_name)}")
    print(f"      - {_image_path(curves_image_name)}")
    print(f"      - {_image_path(acoustic_image_name)}")
    print(f"  → Model checkpoint: {checkpoint_path / best_checkpoint_name}")

    return model, history


def _get_train_default(param_name: str):
    """Read default values directly from train() signature (single source of truth)."""
    param = inspect.signature(train).parameters.get(param_name)
    if param is None or param.default is inspect._empty:
        raise ValueError(f"Parameter '{param_name}' has no default in train()")
    return param.default


if __name__ == "__main__":
    # ============================================================
    # Run Training with Command Line Arguments
    #
    # Usage with synthetic data (default):
    #   uv run python train.py
    #
    # Usage with experimental data (from transformer.py output):
    #   uv run python train.py --mode file --data_path data
    #
    # With deep model and anti-overfitting:
    #   uv run python train.py --mode file --data_path data --model deep --epochs 200 --lr 0.0005 --patience 25
    #
    # With unsupervised autoencoder (input reconstruction):
    #   uv run python train.py --mode file --data_path data --model unsupervised_deep --epochs 100
    # ============================================================

    import argparse

    parser = argparse.ArgumentParser(
        description="Train Ultrasonic Signal Denoising CAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    parser.add_argument(
        "--mode",
        type=str,
        default=_get_train_default("data_mode"),
        choices=["synthetic", "file"],
        help="Data mode: synthetic or file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=_get_train_default("data_path"),
        help="Path to data directory (for file mode)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=_get_train_default("num_epochs"),
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=_get_train_default("batch_size"),
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_get_train_default("learning_rate"),
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=_get_train_default("early_stopping_patience"),
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=_get_train_default("min_epochs"),
        help="Minimum epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=_get_train_default("dropout_rate"),
        help="Dropout rate for regularization (0.1 recommended, 0.4 was too aggressive)",
    )

    # Data augmentation
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (time flip, amplitude scaling, noise injection, time shift)",
    )

    # Synthetic mode parameters
    parser.add_argument(
        "--num_train",
        type=int,
        default=_get_train_default("num_train"),
        help="Number of synthetic training samples",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=_get_train_default("num_val"),
        help="Number of synthetic validation samples",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default=_get_train_default("model_type"),
        choices=["lightweight", "deeper", "deep", "unsupervised_deep"],
        help="Model type: lightweight (3层), deeper (4层), deep (5层), unsupervised_deep (无监督5层)",
    )
    parser.add_argument(
        "--deeper", action="store_true", help="(Deprecated) Use --model deeper instead"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_get_train_default("seed"),
        help="Random seed (default: random each run; set integer for reproducibility)",
    )

    args = parser.parse_args()

    # Handle legacy --deeper flag
    model_type = args.model
    if args.deeper and model_type == "lightweight":
        model_type = "deeper"

    model, history = train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train=args.num_train,
        num_val=args.num_val,
        save_best=True,
        model_type=model_type,
        seed=args.seed,
        data_mode=args.mode,
        data_path=args.data_path,
        early_stopping_patience=args.patience,
        min_epochs=args.min_epochs,
        dropout_rate=args.dropout,
        augment=args.augment,
    )
