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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm

from data_utils import UltrasonicDataset, create_dataloaders
from model import LightweightCAE, DeeperCAE, DeepCAE, count_parameters


def calculate_psnr(
    clean: torch.Tensor, 
    denoised: torch.Tensor, 
    max_val: float = 1.0
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
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


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
    signal_power = torch.mean(signal ** 2).item()
    noise_power = torch.mean(noise ** 2).item()
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def plot_pre_training_samples(
    dataloader: torch.utils.data.DataLoader,
    save_path: str = "fig_pre_train_samples.png",
    num_samples: int = 3
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
    fig.suptitle('Pre-Training Samples: Noisy Input vs Clean Ground Truth', 
                 fontsize=14, fontweight='bold')
    
    # Time axis (in microseconds for display)
    time_us = np.linspace(0, 160, 1000)  # 160 μs duration
    
    for i in range(num_samples):
        noisy_signal = noisy[i, 0].numpy()
        clean_signal = clean[i, 0].numpy()
        
        # Calculate input SNR
        noise = noisy_signal - clean_signal
        snr = calculate_snr(torch.from_numpy(clean_signal), torch.from_numpy(noise))
        
        # Plot noisy signal
        axes[i, 0].plot(time_us, noisy_signal, 'b-', linewidth=0.7, alpha=0.8)
        axes[i, 0].set_title(f'Sample {i+1}: Noisy Input (SNR: {snr:.1f} dB)')
        axes[i, 0].set_xlabel('Time (μs)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_ylim(-2.5, 2.5)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        
        # Plot clean signal
        axes[i, 1].plot(time_us, clean_signal, 'g-', linewidth=0.8)
        axes[i, 1].set_title(f'Sample {i+1}: Clean Ground Truth')
        axes[i, 1].set_xlabel('Time (μs)')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_ylim(-2.5, 2.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved pre-training samples to {save_path}")


def plot_results(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = "fig_results.png",
    num_samples: int = 6,
    train_config: dict = None
) -> None:
    """
    Plot denoising results after training.
    
    Shows 3 rows (Noisy, Clean, Denoised) x num_samples columns.
    Each column represents one signal sample.
    
    Args:
        model: Trained CAE model
        dataloader: DataLoader for test samples
        device: Device to run inference on
        save_path: Path to save the figure
        num_samples: Number of samples to visualize (default 6)
        train_config: Dictionary of training configuration to display
    """
    model.eval()
    
    # Collect multiple batches to get more samples for random selection
    all_noisy = []
    all_clean = []
    for batch_noisy, batch_clean in dataloader:
        all_noisy.append(batch_noisy)
        all_clean.append(batch_clean)
        if len(all_noisy) * batch_noisy.shape[0] >= 100:  # Collect ~100 samples
            break
    
    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    
    # Randomly select samples
    total_samples = all_noisy.shape[0]
    random_indices = np.random.choice(total_samples, size=min(num_samples, total_samples), replace=False)
    
    noisy = all_noisy[random_indices]
    clean = all_clean[random_indices]
    noisy_device = noisy.to(device)
    
    # Run inference
    with torch.no_grad():
        denoised = model(noisy_device)
    
    # Move to CPU for plotting
    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()
    denoised_np = denoised.cpu().numpy()
    
    # Time axis (microseconds)
    time_us = np.linspace(0, 160, noisy_np.shape[-1])
    
    # Create figure: 3 rows (Noisy, Clean, Denoised) x num_samples columns
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))
    
    # Build title with training config
    title = 'Denoising Results: Noisy Input → Clean Ground Truth → Denoised Output'
    if train_config:
        config_str = f"model={train_config.get('model', 'N/A')}, " \
                     f"epochs={train_config.get('epochs', 'N/A')}, " \
                     f"dropout={train_config.get('dropout', 'N/A')}, " \
                     f"augment={train_config.get('augment', False)}, " \
                     f"mode={train_config.get('mode', 'N/A')}"
        if train_config.get('best_psnr'):
            config_str += f", best_val_psnr={train_config['best_psnr']:.2f}dB"
        title = f'{title}\n[{config_str}]'
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    row_titles = ['Noisy Input', 'Clean Ground Truth', 'Denoised Output']
    colors = ['blue', 'green', 'red']
    
    for col in range(num_samples):
        noisy_sig = noisy_np[col, 0]
        clean_sig = clean_np[col, 0]
        denoised_sig = denoised_np[col, 0]
        signals = [noisy_sig, clean_sig, denoised_sig]
        
        # Calculate metrics
        psnr = calculate_psnr(
            torch.from_numpy(clean_np[col]), 
            torch.from_numpy(denoised_np[col])
        )
        
        # Input SNR
        noise = noisy_sig - clean_sig
        input_snr = calculate_snr(
            torch.from_numpy(clean_sig), 
            torch.from_numpy(noise)
        )
        
        for row in range(3):
            ax = axes[row, col] if num_samples > 1 else axes[row]
            ax.plot(time_us, signals[row], color=colors[row], linewidth=0.7)
            
            # Add title with metrics
            if row == 0:
                ax.set_title(f'Sample {col+1}\n{row_titles[row]}\n(Input SNR: {input_snr:.1f} dB)', fontsize=10)
            elif row == 2:
                ax.set_title(f'{row_titles[row]}\n(PSNR: {psnr:.2f} dB)', fontsize=10)
            else:
                ax.set_title(row_titles[row], fontsize=10)
            
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Amplitude')
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved results to {save_path}")


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = "fig_training_curves.png"
) -> None:
    """
    Plot training and validation loss/PSNR curves.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_psnr', 'val_psnr'
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # PSNR curve
    axes[1].plot(epochs, history['train_psnr'], 'b-', label='Train PSNR', linewidth=1.5)
    axes[1].plot(epochs, history['val_psnr'], 'r-', label='Val PSNR', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved training curves to {save_path}")


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function (MSE)
        optimizer: Optimizer (Adam)
        device: Device to train on
        
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
        
        # Forward pass
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        
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
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on validation set.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
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
            
            denoised = model(noisy)
            loss = criterion(denoised, clean)
            
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
    checkpoint_dir: str = "checkpoints",
    use_deeper_model: bool = False,
    model_type: str = 'lightweight',  # 'lightweight', 'deeper', 'deep'
    seed: int = 42,
    data_mode: str = 'synthetic',
    data_path: str = None,
    early_stopping_patience: int = 50,  # More patient: wait 50 epochs without improvement
    min_epochs: int = 30,  # Minimum epochs before early stopping can trigger
    dropout_rate: float = 0.1,  # Lower dropout (0.1) - 0.4 was too aggressive
    augment: bool = False  # Enable data augmentation for training set
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
        model_type: Model architecture - 'lightweight' (3层), 'deeper' (4层), 'deep' (5层)
        seed: Random seed for reproducibility
        data_mode: 'synthetic' or 'file'
        data_path: Path to data directory (for file mode, output of transformer.py)
        early_stopping_patience: Stop training if validation PSNR doesn't improve for N epochs
        dropout_rate: Dropout probability for model regularization (higher = more regularization)
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    if data_mode == 'file':
        print(f"\n[INFO] Loading experimental data from {data_path}...")
        train_loader, val_loader = create_dataloaders(
            batch_size=batch_size,
            seed=seed,
            mode='file',
            data_path=data_path,
            augment=augment
        )
        print(f"[INFO] Data mode: FILE (experimental data)")
        if augment:
            print(f"[INFO] Data augmentation: ENABLED")
    else:
        print("\n[INFO] Creating synthetic datasets...")
        train_loader, val_loader = create_dataloaders(
            num_train=num_train,
            num_val=num_val,
            batch_size=batch_size,
            seed=seed,
            mode='synthetic',
            augment=augment
        )
        print(f"[INFO] Data mode: SYNTHETIC")
        print(f"[INFO] Training samples: {num_train}, Validation samples: {num_val}")
        if augment:
            print(f"[INFO] Data augmentation: ENABLED")
    
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    # Support legacy use_deeper_model parameter
    if use_deeper_model and model_type == 'lightweight':
        model_type = 'deeper'
    
    if model_type == 'deep':
        model = DeepCAE(dropout_rate=dropout_rate).to(device)
        print(f"\n[INFO] Using DeepCAE model (5层, 宽通道)")
    elif model_type == 'deeper':
        model = DeeperCAE(dropout_rate=dropout_rate).to(device)
        print(f"\n[INFO] Using DeeperCAE model (4层)")
    else:
        model = LightweightCAE(dropout_rate=dropout_rate).to(device)
        print(f"\n[INFO] Using LightweightCAE model (3层)")
    
    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    print(f"[INFO] Dropout rate: {dropout_rate}")
    
    # ============================================================
    # Loss Function and Optimizer
    # ============================================================
    criterion = nn.MSELoss()  # Minimizing MSE maximizes PSNR
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Stronger L2 regularization
    
    # Learning rate scheduler: Cosine Annealing with Warm Restarts
    # Better for deep networks - allows escaping local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,              # Initial restart period
        T_mult=2,            # Double the period after each restart
        eta_min=1e-6         # Minimum learning rate
    )
    
    print(f"[INFO] Optimizer: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0=25)")
    print(f"[INFO] Early Stopping: patience={early_stopping_patience}, min_epochs={min_epochs}")
    print(f"[INFO] Loss: MSELoss")
    
    # ============================================================
    # Create Checkpoint Directory
    # ============================================================
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # ============================================================
    # Pre-Training Visualization
    # ============================================================
    print("\n" + "="*60)
    print("[INFO] Generating pre-training samples visualization...")
    print("="*60)
    plot_pre_training_samples(train_loader, "fig_pre_train_samples.png")
    
    # ============================================================
    # Training Loop
    # ============================================================
    print("\n" + "="*60)
    print(f"[INFO] Starting training for {num_epochs} epochs...")
    print("="*60 + "\n")
    
    history: Dict[str, list] = {
        'train_loss': [],
        'val_loss': [],
        'train_psnr': [],
        'val_psnr': []
    }
    
    best_val_psnr = -float('inf')
    early_stopping_counter = 0  # Counter for early stopping
    
    # Progress bar
    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    
    for epoch in pbar:
        # Train one epoch
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        
        # Update learning rate scheduler (CosineAnnealingWarmRestarts uses epoch)
        scheduler.step(epoch)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_psnr'].append(train_psnr)
        history['val_psnr'].append(val_psnr)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_psnr': f'{val_psnr:.2f}dB'
        })
        
        # Print detailed progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_loss': val_loss,
                'train_psnr': train_psnr,
                'train_loss': train_loss,
            }, checkpoint_path / "best_model.pth")
            tqdm.write(f"  → Saved new best model (Val PSNR: {val_psnr:.2f} dB)")
        else:
            early_stopping_counter += 1
            # Only trigger early stopping after minimum epochs
            if epoch >= min_epochs and early_stopping_counter >= early_stopping_patience:
                tqdm.write(f"\n[INFO] Early stopping triggered at epoch {epoch}! No improvement for {early_stopping_patience} epochs.")
                break
    
    # ============================================================
    # Load Best Model
    # ============================================================
    if save_best and (checkpoint_path / "best_model.pth").exists():
        checkpoint = torch.load(
            checkpoint_path / "best_model.pth", 
            map_location=device,
            weights_only=False  # Trust our own checkpoint
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n[INFO] Loaded best model from epoch {checkpoint['epoch']} "
              f"(Val PSNR: {checkpoint['val_psnr']:.2f} dB)")
    
    # ============================================================
    # Post-Training Visualizations
    # ============================================================
    print("\n" + "="*60)
    print("[INFO] Generating post-training visualizations...")
    print("="*60)
    
    # Build training config for display
    train_config = {
        'model': model_type,
        'epochs': num_epochs,
        'dropout': dropout_rate,
        'augment': augment,
        'mode': data_mode,
        'best_psnr': best_val_psnr
    }
    
    # Results visualization
    plot_results(model, val_loader, device, "fig_results.png", train_config=train_config)
    
    # Training curves
    plot_training_curves(history, "fig_training_curves.png")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("[INFO] Training Complete!")
    print("="*60)
    print(f"  → Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"  → Final training PSNR: {history['train_psnr'][-1]:.2f} dB")
    print(f"  → Figures saved:")
    print(f"      - fig_pre_train_samples.png")
    print(f"      - fig_results.png")
    print(f"      - fig_training_curves.png")
    print(f"  → Model checkpoint: {checkpoint_path / 'best_model.pth'}")
    
    return model, history


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
    # ============================================================
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Ultrasonic Signal Denoising CAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source
    parser.add_argument('--mode', type=str, default='synthetic',
                        choices=['synthetic', 'file'],
                        help='Data mode: synthetic or file')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to data directory (for file mode)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_epochs', type=int, default=30,
                        help='Minimum epochs before early stopping can trigger')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization (0.1 recommended, 0.4 was too aggressive)')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (time flip, amplitude scaling, noise injection, time shift)')
    
    # Synthetic mode parameters
    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of synthetic training samples')
    parser.add_argument('--num_val', type=int, default=1000,
                        help='Number of synthetic validation samples')
    
    # Model options
    parser.add_argument('--model', type=str, default='lightweight',
                        choices=['lightweight', 'deeper', 'deep'],
                        help='Model type: lightweight (3层), deeper (4层), deep (5层)')
    parser.add_argument('--deeper', action='store_true',
                        help='(Deprecated) Use --model deeper instead')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Handle legacy --deeper flag
    model_type = args.model
    if args.deeper and model_type == 'lightweight':
        model_type = 'deeper'
    
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
        augment=args.augment
    )
