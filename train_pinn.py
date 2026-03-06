"""
PINN Training Pipeline for Ultrasonic Signal Denoising

Physics-Informed Neural Network (PINN) variant of the DeepCAE training,
embedding 1D acoustic wave equation constraint into the loss function:

    L_total = L_data + λ · L_physics
    
    L_data    = MSE(denoised, clean)
    L_physics = mean(|∂²u/∂t²|²)   — wave equation residual

Usage:
    uv run python train_pinn.py
    uv run python train_pinn.py --mode file --data_path data --physics_weight 0.001
    uv run python train_pinn.py --model deep --epochs 100 --physics_weight 0.01
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

from data_utils import create_dataloaders
from model import DeepCAE_PINN, count_parameters
from train import (
    calculate_psnr,
    calculate_snr,
    plot_pre_training_samples,
    plot_results,
)
from acoustic_validation import run_acoustic_validation


# ============================================================
# PINN Loss Function
# ============================================================

class PINNLoss(nn.Module):
    """
    Combined data + physics loss for PINN training.
    
    L_total = L_data + physics_weight · L_physics
    
    Where:
        L_data    = MSE(denoised, clean)
        L_physics = mean(|wave_equation_residual|²)
    """
    
    def __init__(self, physics_weight: float = 0.001):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_weight = physics_weight
    
    def forward(
        self,
        denoised: torch.Tensor,
        clean: torch.Tensor,
        physics_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            denoised: Model output (B, 1, 1000)
            clean: Ground truth (B, 1, 1000)
            physics_residual: Wave equation residual (B, 1, 998)
            
        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        data_loss = self.data_loss_fn(denoised, clean)
        
        # Normalize physics residual to avoid scale issues with large d²u/dt²
        # Use mean squared value of the residual
        physics_loss = torch.mean(physics_residual ** 2)
        
        total_loss = data_loss + self.physics_weight * physics_loss
        
        return total_loss, data_loss, physics_loss


# ============================================================
# PINN Training & Validation
# ============================================================

def train_epoch_pinn(
    model: DeepCAE_PINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: PINNLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch with PINN loss.
    
    Returns:
        Tuple of (avg_total_loss, avg_data_loss, avg_physics_loss, avg_psnr)
    """
    model.train()
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    psnr_sum = 0.0
    num_batches = 0
    
    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # PINN forward pass: get denoised output + physics residual
        denoised, residual = model.physics_forward(noisy)
        
        # Compute combined loss
        total_loss, data_loss, physics_loss = criterion(denoised, clean, residual)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss_sum += total_loss.item()
        data_loss_sum += data_loss.item()
        physics_loss_sum += physics_loss.item()
        psnr_sum += calculate_psnr(clean, denoised)
        num_batches += 1
    
    return (
        total_loss_sum / num_batches,
        data_loss_sum / num_batches,
        physics_loss_sum / num_batches,
        psnr_sum / num_batches,
    )


def validate_pinn(
    model: DeepCAE_PINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: PINNLoss,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Validate model with PINN loss.
    
    Returns:
        Tuple of (avg_total_loss, avg_data_loss, avg_physics_loss, avg_psnr)
    """
    model.eval()
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    psnr_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            denoised, residual = model.physics_forward(noisy)
            total_loss, data_loss, physics_loss = criterion(denoised, clean, residual)
            
            total_loss_sum += total_loss.item()
            data_loss_sum += data_loss.item()
            physics_loss_sum += physics_loss.item()
            psnr_sum += calculate_psnr(clean, denoised)
            num_batches += 1
    
    return (
        total_loss_sum / num_batches,
        data_loss_sum / num_batches,
        physics_loss_sum / num_batches,
        psnr_sum / num_batches,
    )


# ============================================================
# PINN Training Curves Visualization
# ============================================================

def plot_pinn_training_curves(
    history: Dict[str, list],
    save_path: str = "fig_pinn_training_curves.png",
) -> None:
    """
    Plot PINN training curves including physics loss.
    
    Shows 3 subplots:
        1. Total Loss (train vs val)
        2. Data Loss + Physics Loss decomposition
        3. PSNR curves
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PINN Training Progress', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # 1. Total loss
    axes[0].plot(epochs, history['train_total_loss'], 'b-', label='Train Total', linewidth=1.5)
    axes[0].plot(epochs, history['val_total_loss'], 'r-', label='Val Total', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss (Data + λ·Physics)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 2. Loss decomposition
    axes[1].plot(epochs, history['train_data_loss'], 'b-', label='Train Data Loss', linewidth=1.5)
    axes[1].plot(epochs, history['val_data_loss'], 'r-', label='Val Data Loss', linewidth=1.5)
    axes[1].plot(epochs, history['train_physics_loss'], 'b--', label='Train Physics Loss', linewidth=1.0, alpha=0.7)
    axes[1].plot(epochs, history['val_physics_loss'], 'r--', label='Val Physics Loss', linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Decomposition')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # 3. PSNR
    axes[2].plot(epochs, history['train_psnr'], 'b-', label='Train PSNR', linewidth=1.5)
    axes[2].plot(epochs, history['val_psnr'], 'r-', label='Val PSNR', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('PSNR Curve')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved PINN training curves to {save_path}")


# ============================================================
# Main PINN Training Function
# ============================================================

def train_pinn(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_train: int = 5000,
    num_val: int = 1000,
    save_best: bool = True,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
    data_mode: str = 'synthetic',
    data_path: str = None,
    early_stopping_patience: int = 50,
    min_epochs: int = 30,
    dropout_rate: float = 0.1,
    augment: bool = False,
    # PINN-specific parameters
    physics_weight: float = 0.001,
    wave_speed: float = 5900.0,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Main PINN training function.
    
    Same structure as train() in train.py, but uses:
    - DeepCAE_PINN model with physics residual computation
    - PINNLoss combining data + physics losses
    - Extended logging for physics loss tracking
    
    Args:
        physics_weight: Weight λ for physics loss (higher = stronger constraint)
        wave_speed: Speed of sound in medium (m/s), default 5900 for steel
        (all other args same as train.train())
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set seeds
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
            augment=augment,
        )
        print(f"[INFO] Data mode: FILE (experimental data)")
    else:
        print("\n[INFO] Creating synthetic datasets...")
        train_loader, val_loader = create_dataloaders(
            num_train=num_train,
            num_val=num_val,
            batch_size=batch_size,
            seed=seed,
            mode='synthetic',
            augment=augment,
        )
        print(f"[INFO] Data mode: SYNTHETIC")
        print(f"[INFO] Training samples: {num_train}, Validation samples: {num_val}")
    
    if augment:
        print(f"[INFO] Data augmentation: ENABLED")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ============================================================
    # Initialize PINN Model
    # ============================================================
    model = DeepCAE_PINN(
        dropout_rate=dropout_rate,
        wave_speed=wave_speed,
    ).to(device)
    
    print(f"\n[INFO] Using DeepCAE_PINN model (Physics-Informed)")
    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    print(f"[INFO] Dropout rate: {dropout_rate}")
    print(f"[INFO] Physics weight (λ): {physics_weight}")
    print(f"[INFO] Wave speed: {wave_speed} m/s")
    print(f"[INFO] Time step (dt): {model.dt:.2e} s")
    
    # ============================================================
    # Loss Function and Optimizer
    # ============================================================
    criterion = PINNLoss(physics_weight=physics_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=1e-6,
    )
    
    print(f"[INFO] Optimizer: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"[INFO] Scheduler: CosineAnnealingWarmRestarts (T_0=25)")
    print(f"[INFO] Early Stopping: patience={early_stopping_patience}, min_epochs={min_epochs}")
    print(f"[INFO] Loss: MSE + {physics_weight} × PhysicsLoss")
    
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
    plot_pre_training_samples(train_loader, "fig_pinn_pre_train_samples.png")
    
    # ============================================================
    # Training Loop
    # ============================================================
    print("\n" + "="*60)
    print(f"[INFO] Starting PINN training for {num_epochs} epochs...")
    print("="*60 + "\n")
    
    history: Dict[str, list] = {
        'train_total_loss': [],
        'train_data_loss': [],
        'train_physics_loss': [],
        'train_psnr': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_physics_loss': [],
        'val_psnr': [],
    }
    
    best_val_psnr = -float('inf')
    early_stopping_counter = 0
    
    pbar = tqdm(range(1, num_epochs + 1), desc="PINN Training", unit="epoch")
    
    for epoch in pbar:
        # Train one epoch
        train_total, train_data, train_phys, train_psnr = train_epoch_pinn(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_total, val_data, val_phys, val_psnr = validate_pinn(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(epoch)
        
        # Record history
        history['train_total_loss'].append(train_total)
        history['train_data_loss'].append(train_data)
        history['train_physics_loss'].append(train_phys)
        history['train_psnr'].append(train_psnr)
        history['val_total_loss'].append(val_total)
        history['val_data_loss'].append(val_data)
        history['val_physics_loss'].append(val_phys)
        history['val_psnr'].append(val_psnr)
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{train_total:.6f}',
            'phys': f'{train_phys:.2e}',
            'val_psnr': f'{val_psnr:.2f}dB',
        })
        
        # Detailed progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            gap = train_psnr - val_psnr
            tqdm.write(
                f"Epoch [{epoch:3d}/{num_epochs}] | "
                f"Total: {train_total:.6f} | Data: {train_data:.6f} | "
                f"Phys: {train_phys:.2e} | "
                f"Val PSNR: {val_psnr:.2f} dB | Gap: {gap:.2f} dB | "
                f"LR: {current_lr:.2e}"
            )
        
        # Save best model and early stopping
        if save_best and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_total_loss': val_total,
                'val_data_loss': val_data,
                'val_physics_loss': val_phys,
                'train_psnr': train_psnr,
                'physics_weight': physics_weight,
                'wave_speed': wave_speed,
            }, checkpoint_path / "best_pinn_model.pth")
            tqdm.write(f"  → Saved new best PINN model (Val PSNR: {val_psnr:.2f} dB)")
        else:
            early_stopping_counter += 1
            if epoch >= min_epochs and early_stopping_counter >= early_stopping_patience:
                tqdm.write(
                    f"\n[INFO] Early stopping triggered at epoch {epoch}! "
                    f"No improvement for {early_stopping_patience} epochs."
                )
                break
    
    # ============================================================
    # Load Best Model
    # ============================================================
    if save_best and (checkpoint_path / "best_pinn_model.pth").exists():
        checkpoint = torch.load(
            checkpoint_path / "best_pinn_model.pth",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n[INFO] Loaded best PINN model from epoch {checkpoint['epoch']} "
              f"(Val PSNR: {checkpoint['val_psnr']:.2f} dB)")
    
    # ============================================================
    # Post-Training Visualizations
    # ============================================================
    print("\n" + "="*60)
    print("[INFO] Generating post-training visualizations...")
    print("="*60)
    
    train_config = {
        'model': 'DeepCAE_PINN',
        'epochs': num_epochs,
        'dropout': dropout_rate,
        'augment': augment,
        'mode': data_mode,
        'best_psnr': best_val_psnr,
    }
    
    # Results visualization (reuse from train.py)
    plot_results(model, val_loader, device, "fig_pinn_results.png", train_config=train_config)
    
    # PINN-specific training curves (includes physics loss)
    plot_pinn_training_curves(history, "fig_pinn_training_curves.png")
    
    # Acoustic feature validation (声学特征验证)
    print("\n" + "="*60)
    print("[INFO] Running acoustic feature validation...")
    print("="*60)
    run_acoustic_validation(model, val_loader, device, save_path="fig_pinn_acoustic_validation.png")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("[INFO] PINN Training Complete!")
    print("="*60)
    print(f"  → Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"  → Final training PSNR: {history['train_psnr'][-1]:.2f} dB")
    print(f"  → Final physics loss: {history['train_physics_loss'][-1]:.2e}")
    print(f"  → Physics weight (λ): {physics_weight}")
    print(f"  → Figures saved:")
    print(f"      - fig_pinn_pre_train_samples.png")
    print(f"      - fig_pinn_results.png")
    print(f"      - fig_pinn_training_curves.png")
    print(f"      - fig_pinn_acoustic_validation.png")
    print(f"  → Model checkpoint: {checkpoint_path / 'best_pinn_model.pth'}")
    
    return model, history


if __name__ == "__main__":
    # ============================================================
    # PINN Training with Command Line Arguments
    # 
    # Usage with synthetic data:
    #   uv run python train_pinn.py
    #
    # Usage with experimental data:
    #   uv run python train_pinn.py --mode file --data_path data
    #
    # Tune physics weight:
    #   uv run python train_pinn.py --physics_weight 0.01
    # ============================================================
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train PINN for Ultrasonic Signal Denoising',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                        help='Early stopping patience')
    parser.add_argument('--min_epochs', type=int, default=30,
                        help='Minimum epochs before early stopping')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    
    # Synthetic mode parameters
    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of synthetic training samples')
    parser.add_argument('--num_val', type=int, default=1000,
                        help='Number of synthetic validation samples')
    
    # Model options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # PINN-specific parameters
    parser.add_argument('--physics_weight', type=float, default=0.001,
                        help='Weight λ for physics loss (higher = stronger constraint)')
    parser.add_argument('--wave_speed', type=float, default=5900.0,
                        help='Speed of sound in medium (m/s), 5900 for steel')
    
    args = parser.parse_args()
    
    model, history = train_pinn(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train=args.num_train,
        num_val=args.num_val,
        save_best=True,
        seed=args.seed,
        data_mode=args.mode,
        data_path=args.data_path,
        early_stopping_patience=args.patience,
        min_epochs=args.min_epochs,
        dropout_rate=args.dropout,
        augment=args.augment,
        physics_weight=args.physics_weight,
        wave_speed=args.wave_speed,
    )
