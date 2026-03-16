"""
DeepSets + PINN Training Pipeline for Ultrasonic Signal Denoising

Implements training with the 2D acoustic wave equation constraint:

    L_total = L_data + λ · L_physics

    L_data    = MSE(denoised, clean)
    L_physics = mean(|u_tt − c²(u_xx + u_yy)|² / ω₀⁴)

Usage:
    uv run python train_deepsets_pinn.py
    uv run python train_deepsets_pinn.py --data_path data --physics_weight 0.001
    uv run python train_deepsets_pinn.py --epochs 100 --patch_size 7
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
from pathlib import Path
from typing import Tuple, Dict, Any, cast
from tqdm import tqdm

from model.model_deepsets import DeepSetsPINN, SpatialAuxiliaryCAE, count_parameters
from data import create_deepsets_dataloaders, GRID_SPACING
from scripts.analysis import acoustic_validation as av


RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def _image_path(filename: str) -> str:
    """Build a default image output path under results/images."""
    return str(IMAGES_DIR / filename)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for an output file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_image_dir(checkpoint_dir: str) -> Path:
    """Resolve image output directory from checkpoint directory."""
    ckpt_path = Path(checkpoint_dir)
    # Unified pipeline passes <run_dir>/checkpoints. Keep standalone compatibility.
    if ckpt_path.name == "checkpoints":
        return ckpt_path.parent / "images"
    return RESULTS_DIR / "images"


# ============================================================
# PSNR Calculation (adapted from train.py)
# ============================================================


def calculate_psnr(
    clean: torch.Tensor,
    denoised: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB.

    PSNR = 10 · log10(MAX² / MSE)

    Args:
        clean: Ground truth tensor
        denoised: Reconstructed tensor
        max_val: Maximum possible signal value (1.0 for normalised)

    Returns:
        PSNR in dB
    """
    mse = torch.mean((clean - denoised) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def calculate_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = torch.mean(signal**2).item()
    noise_power = torch.mean(noise**2).item()
    if noise_power < 1e-10:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# ============================================================
# PINN Loss Function
# ============================================================


class DeepSetsPINNLoss(nn.Module):
    """
    Combined data + wave equation physics loss.

    L_total = L_data + physics_weight · L_physics

    Where:
        L_data    = MSE(denoised, clean)     — per-signal reconstruction
        L_physics = mean(residual²)          — 2D wave equation residual
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
        Args:
            denoised: (B, R, T) model output
            clean:    (B, R, T) ground truth
            physics_residual: (N_interior, T-2) wave equation residual

        Returns:
            (total_loss, data_loss, physics_loss)
        """
        data_loss = self.data_loss_fn(denoised, clean)
        physics_loss = torch.mean(physics_residual**2)
        total_loss = data_loss + self.physics_weight * physics_loss
        return total_loss, data_loss, physics_loss


# ============================================================
# Training & Validation
# ============================================================


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: DeepSetsPINNLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    grid_cols: int = 41,
    grid_rows: int = 41,
) -> Tuple[float, float, float, float]:
    """
    Train one epoch with DeepSets PINN loss.

    Returns:
        (avg_total_loss, avg_data_loss, avg_physics_loss, avg_psnr)
    """
    model.train()
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    psnr_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        noisy = batch["noisy_signals"].to(device)  # (B, R, T)
        clean = batch["clean_signals"].to(device)  # (B, R, T)
        coords = batch["coordinates"].to(device)  # (B, R, 2)
        grid_idx = batch["grid_indices"].to(device)  # (B, R, 2)

        # Forward with physics
        denoised, residual = model.physics_forward(
            noisy, coords, grid_idx, grid_cols, grid_rows
        )

        # Combined loss
        total_loss, data_loss, physics_loss = criterion(denoised, clean, residual)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to stabilise physics loss gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: DeepSetsPINNLoss,
    device: torch.device,
    grid_cols: int = 41,
    grid_rows: int = 41,
) -> Tuple[float, float, float, float]:
    """
    Validate with DeepSets PINN loss.

    Returns:
        (avg_total_loss, avg_data_loss, avg_physics_loss, avg_psnr)
    """
    model.eval()
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    psnr_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            noisy = batch["noisy_signals"].to(device)
            clean = batch["clean_signals"].to(device)
            coords = batch["coordinates"].to(device)
            grid_idx = batch["grid_indices"].to(device)

            denoised, residual = model.physics_forward(
                noisy, coords, grid_idx, grid_cols, grid_rows
            )
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
# Training Curves Visualisation
# ============================================================


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = _image_path("fig_deepsets_pinn_training_curves.png"),
) -> None:
    """
    Plot 3-panel training curves: total loss, loss decomposition, PSNR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DeepSets PINN Training Progress", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_total_loss"]) + 1)

    # 1. Total loss
    axes[0].plot(
        epochs, history["train_total_loss"], "b-", label="Train Total", linewidth=1.5
    )
    axes[0].plot(
        epochs, history["val_total_loss"], "r-", label="Val Total", linewidth=1.5
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (Data + λ·Physics)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # 2. Loss decomposition
    axes[1].plot(
        epochs, history["train_data_loss"], "b-", label="Train Data", linewidth=1.5
    )
    axes[1].plot(
        epochs, history["val_data_loss"], "r-", label="Val Data", linewidth=1.5
    )
    axes[1].plot(
        epochs,
        history["train_physics_loss"],
        "b--",
        label="Train Physics",
        linewidth=1.0,
        alpha=0.7,
    )
    axes[1].plot(
        epochs,
        history["val_physics_loss"],
        "r--",
        label="Val Physics",
        linewidth=1.0,
        alpha=0.7,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Decomposition")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    # 3. PSNR
    axes[2].plot(epochs, history["train_psnr"], "b-", label="Train PSNR", linewidth=1.5)
    axes[2].plot(epochs, history["val_psnr"], "r-", label="Val PSNR", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_title("PSNR Curve")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved training curves to {save_path}")


def plot_sample_results(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = _image_path("fig_deepsets_pinn_results.png"),
    n_samples: int = 6,
) -> None:
    """
    Plot denoising results in a 3-row x n_samples-column layout.

    Rows are: Noisy Input, Clean Ground Truth, Denoised Output.
    Uses the centre signal in each patch for display.
    """
    model.eval()

    all_noisy = []
    all_clean = []
    all_coords = []
    for batch in dataloader:
        all_noisy.append(batch["noisy_signals"])
        all_clean.append(batch["clean_signals"])
        all_coords.append(batch["coordinates"])
        if len(all_noisy) * batch["noisy_signals"].shape[0] >= 100:
            break

    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    all_coords = torch.cat(all_coords, dim=0)

    total_samples = all_noisy.shape[0]
    n_samples = min(n_samples, total_samples)
    random_indices = np.random.choice(total_samples, size=n_samples, replace=False)

    noisy = all_noisy[random_indices]
    clean = all_clean[random_indices]
    coords = all_coords[random_indices]

    with torch.no_grad():
        denoised = model(noisy.to(device), coords.to(device))

    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()
    denoised_np = denoised.cpu().numpy()

    time_us = np.linspace(0, 160, noisy_np.shape[-1])
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 10))

    fig.suptitle(
        "Denoising Results: Noisy Input → Clean Ground Truth → Denoised Output",
        fontsize=12,
        fontweight="bold",
    )

    row_titles = ["Noisy Input", "Clean Ground Truth", "Denoised Output"]
    colors = ["blue", "green", "red"]

    for col in range(n_samples):
        centre = noisy_np[col].shape[0] // 2
        noisy_sig = noisy_np[col, centre]
        clean_sig = clean_np[col, centre]
        denoised_sig = denoised_np[col, centre]
        signals = [noisy_sig, clean_sig, denoised_sig]

        psnr = calculate_psnr(
            torch.from_numpy(clean_np[col, centre : centre + 1]),
            torch.from_numpy(denoised_np[col, centre : centre + 1]),
        )
        input_noise = noisy_sig - clean_sig
        input_snr = calculate_snr(
            torch.from_numpy(clean_sig), torch.from_numpy(input_noise)
        )

        for row in range(3):
            ax = axes[row, col] if n_samples > 1 else axes[row]
            ax.plot(time_us, signals[row], color=colors[row], linewidth=0.7)

            if row == 0:
                ax.set_title(
                    f"Sample {col + 1}\n{row_titles[row]}\n(Input SNR: {input_snr:.1f} dB)",
                    fontsize=10,
                )
            elif row == 2:
                ax.set_title(f"{row_titles[row]}\n(PSNR: {psnr:.2f} dB)", fontsize=10)
            else:
                ax.set_title(row_titles[row], fontsize=10)

            ax.set_xlabel("Time (μs)")
            ax.set_ylabel("Amplitude")
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved sample results to {save_path}")


def plot_pre_training_samples_deepsets(
    dataloader: torch.utils.data.DataLoader,
    save_path: str = _image_path("fig_deepsets_pinn_pre_train_samples.png"),
    num_samples: int = 3,
) -> None:
    """
    Plot noisy vs clean centre signals before training starts.
    """
    batch = next(iter(dataloader))
    noisy = batch["noisy_signals"]
    clean = batch["clean_signals"]

    num_samples = min(num_samples, noisy.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Pre-Training Samples: Noisy Input vs Clean Ground Truth",
        fontsize=14,
        fontweight="bold",
    )

    time_us = np.linspace(0, 160, noisy.shape[-1])

    for i in range(num_samples):
        centre = noisy.shape[1] // 2
        noisy_signal = noisy[i, centre].cpu().numpy()
        clean_signal = clean[i, centre].cpu().numpy()

        noise = noisy_signal - clean_signal
        snr = calculate_snr(torch.from_numpy(clean_signal), torch.from_numpy(noise))

        axes[i, 0].plot(time_us, noisy_signal, "b-", linewidth=0.7, alpha=0.8)
        axes[i, 0].set_title(f"Sample {i + 1}: Noisy Input (SNR: {snr:.1f} dB)")
        axes[i, 0].set_xlabel("Time (μs)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_ylim(-2.5, 2.5)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

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


def run_deepsets_acoustic_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = _image_path("fig_deepsets_pinn_acoustic_validation.png"),
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Run acoustic feature validation for DeepSets-style dataloader/model."""
    model.eval()

    all_noisy = []
    all_clean = []
    all_coords = []
    for batch in dataloader:
        all_noisy.append(batch["noisy_signals"])
        all_clean.append(batch["clean_signals"])
        all_coords.append(batch["coordinates"])
        if sum(x.shape[0] for x in all_noisy) >= 100:
            break

    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    all_coords = torch.cat(all_coords, dim=0)

    total = all_noisy.shape[0]
    n = min(num_samples, total)
    indices = np.random.choice(total, size=n, replace=False)

    noisy = all_noisy[indices]
    clean = all_clean[indices]
    coords = all_coords[indices]

    with torch.no_grad():
        predicted = model(noisy.to(device), coords.to(device))

    centre = noisy.shape[1] // 2
    inp_np = noisy.cpu().numpy()[:, centre, :]
    tgt_np = clean.cpu().numpy()[:, centre, :]
    pred_np = predicted.cpu().numpy()[:, centre, :]

    print(f"[INFO] Analyzing {n} samples for acoustic validation...")

    all_features = []
    for i in range(n):
        all_features.append(
            {
                "input": av._extract_all_features(inp_np[i]),
                "target": av._extract_all_features(tgt_np[i]),
                "predicted": av._extract_all_features(pred_np[i]),
            }
        )

    xcorr_tp_list = []
    env_corr_tp_list = []
    coherence_tp_list = []
    arrival_errors = []
    peak_pres_list = []
    rms_pres_list = []
    dom_freq_pres_list = []
    energy_match_list = []

    for i in range(n):
        tgt_sig = tgt_np[i]
        pred_sig = pred_np[i]

        peak_corr, _ = av._cross_correlation_peak(tgt_sig, pred_sig)
        xcorr_tp_list.append(peak_corr)
        env_corr_tp_list.append(av._envelope_correlation(tgt_sig, pred_sig))

        freq_coh, coh = av._spectral_coherence(tgt_sig, pred_sig)
        signal_band = (freq_coh >= 100e3) & (freq_coh <= 500e3)
        avg_coh = float(np.mean(coh[signal_band])) if np.any(signal_band) else 0.0
        coherence_tp_list.append(avg_coh)

        fa = all_features[i]
        arr_tgt = fa["target"]["arrival_time_us"]
        arr_pred = fa["predicted"]["arrival_time_us"]
        if not np.isnan(arr_tgt) and not np.isnan(arr_pred):
            arrival_errors.append(abs(arr_tgt - arr_pred))

        tgt_peak = fa["target"]["peak_amplitude"]
        pred_peak = fa["predicted"]["peak_amplitude"]
        if tgt_peak > 1e-10 and pred_peak > 1e-10:
            peak_pres_list.append(min(pred_peak / tgt_peak, tgt_peak / pred_peak) * 100)

        tgt_rms = fa["target"]["rms"]
        pred_rms = fa["predicted"]["rms"]
        if tgt_rms > 1e-10 and pred_rms > 1e-10:
            rms_pres_list.append(min(pred_rms / tgt_rms, tgt_rms / pred_rms) * 100)

        tgt_df = fa["target"]["dominant_freq_khz"]
        pred_df = fa["predicted"]["dominant_freq_khz"]
        if tgt_df > 1e-3 and pred_df > 1e-3:
            dom_freq_pres_list.append(min(pred_df / tgt_df, tgt_df / pred_df) * 100)

        tgt_sub = np.array(
            [fa["target"].get(f"energy_ratio_{band}", 0) for band in av.SUB_BAND_LABELS]
        )
        pred_sub = np.array(
            [fa["predicted"].get(f"energy_ratio_{band}", 0) for band in av.SUB_BAND_LABELS]
        )
        denom_cos = np.linalg.norm(tgt_sub) * np.linalg.norm(pred_sub)
        if denom_cos > 1e-10:
            energy_match_list.append(float(np.dot(tgt_sub, pred_sub) / denom_cos) * 100)

    quality_metrics = {
        "per_sample": {
            "xcorr_target_pred": xcorr_tp_list,
            "envelope_corr_target_pred": env_corr_tp_list,
            "avg_coherence_target_pred": coherence_tp_list,
        },
        "averaged": {
            "xcorr_target_pred": float(np.mean(xcorr_tp_list))
            if xcorr_tp_list
            else 0.0,
            "envelope_corr_target_pred": float(np.mean(env_corr_tp_list))
            if env_corr_tp_list
            else 0.0,
            "avg_coherence_target_pred": float(np.mean(coherence_tp_list))
            if coherence_tp_list
            else 0.0,
            "arrival_time_error_us": float(np.mean(arrival_errors))
            if arrival_errors
            else float("nan"),
            "peak_amplitude_preservation": float(np.mean(peak_pres_list))
            if peak_pres_list
            else 0.0,
            "rms_preservation": float(np.mean(rms_pres_list)) if rms_pres_list else 0.0,
            "dominant_freq_preservation": float(np.mean(dom_freq_pres_list))
            if dom_freq_pres_list
            else 0.0,
            "energy_distribution_match": float(np.mean(energy_match_list))
            if energy_match_list
            else 0.0,
        },
    }

    av._plot_validation_figure(
        input_signals=inp_np,
        target_signals=tgt_np,
        predicted_signals=pred_np,
        all_features=cast(Any, all_features),
        quality_metrics=quality_metrics,
        save_path=save_path,
    )
    av._print_report(quality_metrics)

    return {"features": all_features, "quality_metrics": quality_metrics}

# ============================================================
# Main Training Function
# ============================================================


def train_deepsets_pinn(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_best: bool = True,
    checkpoint_dir: str = str(CHECKPOINTS_DIR),
    seed: int = 42,
    data_path: str = "data",
    early_stopping_patience: int = 50,
    min_epochs: int = 30,
    dropout_rate: float = 0.1,
    augment: bool = True,
    # Grid parameters
    grid_cols: int = 41,
    grid_rows: int = 41,
    patch_size: int = 5,
    stride: int = 1,
    # Physics parameters
    physics_weight: float = 0.001,
    wave_speed: float = 5900.0,
    center_frequency: float = 250e3,
    dx: float = GRID_SPACING,
    dy: float = GRID_SPACING,
    # Model parameters
    model_type: str = "spatial_cae",
    base_channels: int = 16,
    coord_dim: int = 64,
    # Kept for backward compat with deepsets
    signal_embed_dim: int = 128,
    coord_embed_dim: int = 64,
    point_dim: int = 128,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Main DeepSets PINN training function.

    Args:
        num_epochs:   Maximum training epochs.
        batch_size:   Batch size (number of patches per step).
        learning_rate: Initial learning rate for Adam.
        save_best:    Whether to save best model checkpoint.
        checkpoint_dir: Directory for checkpoints.
        seed:         Random seed.
        data_path:    Root data directory (must contain train/ and val/).
        early_stopping_patience: Epochs without improvement before stop.
        min_epochs:   Minimum epochs before early stopping activates.
        dropout_rate: Dropout rate in middle Conv/Linear layers.
        grid_cols:    Scanning grid columns.
        grid_rows:    Scanning grid rows.
        patch_size:   Square patch side length for DeepSets input.
        stride:       Patch extraction stride.
        physics_weight: Weight λ for physics loss.
        wave_speed:   Acoustic wave speed (m/s).
        center_frequency: Transducer centre frequency (Hz).
        dx:           Grid x spacing (m).
        dy:           Grid y spacing (m).
        signal_embed_dim: Signal encoder output dim.
        coord_embed_dim:  Coordinate MLP output dim.
        point_dim:    Fused point feature dim.

    Returns:
        (trained_model, training_history)
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
    # Create DataLoaders
    # ============================================================
    print(f"\n[INFO] Loading experimental data from {data_path}...")
    train_loader, val_loader = create_deepsets_dataloaders(
        data_root=data_path,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        dx=dx,
        dy=dy,
        augment=augment,
    )

    # ============================================================
    # Initialize Model
    # ============================================================
    if model_type == "spatial_cae":
        model = SpatialAuxiliaryCAE(
            base_channels=base_channels,
            coord_dim=coord_dim,
            dropout_rate=dropout_rate,
            wave_speed=wave_speed,
            center_frequency=center_frequency,
            dx=dx,
            dy=dy,
            patch_size=patch_size,
        ).to(device)
    else:
        model = DeepSetsPINN(
            signal_embed_dim=signal_embed_dim,
            coord_embed_dim=coord_embed_dim,
            point_dim=point_dim,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
            wave_speed=wave_speed,
            center_frequency=center_frequency,
            dx=dx,
            dy=dy,
            patch_size=patch_size,
        ).to(device)

    wavelength = wave_speed / center_frequency
    print(f"\n[INFO] Model: {model.__class__.__name__} (type={model_type})")
    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    print(
        f"[INFO] Patch size: {patch_size}×{patch_size} ({patch_size**2} elements/set)"
    )
    print(f"[INFO] Dropout: {dropout_rate}")
    print(f"[INFO] Physics weight (λ): {physics_weight}")
    print(f"[INFO] Wave speed: {wave_speed} m/s")
    print(f"[INFO] Centre frequency: {center_frequency / 1e3:.1f} kHz")
    print(f"[INFO] Wavelength: {wavelength * 1e3:.2f} mm")
    print(f"[INFO] Grid spacing: dx={dx * 1e3:.2f} mm, dy={dy * 1e3:.2f} mm")
    print(f"[INFO] Grid:  {grid_cols}×{grid_rows}")
    print(f"[INFO] dt:    {model.dt:.2e} s")

    # ============================================================
    # Loss, Optimizer, Scheduler
    # ============================================================
    criterion = DeepSetsPINNLoss(physics_weight=physics_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    print(f"[INFO] Optimizer: Adam (lr={learning_rate}, wd=1e-4)")
    print(f"[INFO] Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    print(
        f"[INFO] Early stop: patience={early_stopping_patience}, "
        f"min_epochs={min_epochs}"
    )

    # Checkpoint directory
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    image_dir = _resolve_image_dir(checkpoint_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    def run_image_path(filename: str) -> str:
        return str(image_dir / filename)

    # ============================================================
    # Pre-Training Visualization
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] Generating pre-training samples visualization...")
    print("=" * 60)
    plot_pre_training_samples_deepsets(
        train_loader, run_image_path("fig_deepsets_pinn_pre_train_samples.png")
    )

    # ============================================================
    # Training Loop
    # ============================================================
    print("\n" + "=" * 60)
    print(f"[INFO] Starting DeepSets PINN training for {num_epochs} epochs...")
    print("=" * 60 + "\n")

    history: Dict[str, list] = {
        "train_total_loss": [],
        "train_data_loss": [],
        "train_physics_loss": [],
        "train_psnr": [],
        "val_total_loss": [],
        "val_data_loss": [],
        "val_physics_loss": [],
        "val_psnr": [],
    }

    best_val_psnr = -float("inf")
    early_stop_counter = 0

    pbar = tqdm(range(1, num_epochs + 1), desc="DeepSets PINN Training", unit="epoch")

    for epoch in pbar:
        # Train
        tr_total, tr_data, tr_phys, tr_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device, grid_cols, grid_rows
        )

        # Validate
        va_total, va_data, va_phys, va_psnr = validate(
            model, val_loader, criterion, device, grid_cols, grid_rows
        )

        scheduler.step(epoch)

        # Record
        history["train_total_loss"].append(tr_total)
        history["train_data_loss"].append(tr_data)
        history["train_physics_loss"].append(tr_phys)
        history["train_psnr"].append(tr_psnr)
        history["val_total_loss"].append(va_total)
        history["val_data_loss"].append(va_data)
        history["val_physics_loss"].append(va_phys)
        history["val_psnr"].append(va_psnr)

        pbar.set_postfix(
            {
                "total": f"{tr_total:.6f}",
                "phys": f"{tr_phys:.2e}",
                "val_psnr": f"{va_psnr:.2f}dB",
            }
        )

        # Detailed log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            gap = tr_psnr - va_psnr
            tqdm.write(
                f"Epoch [{epoch:3d}/{num_epochs}] | "
                f"Total: {tr_total:.6f} | Data: {tr_data:.6f} | "
                f"Phys: {tr_phys:.2e} | "
                f"Val PSNR: {va_psnr:.2f} dB | Gap: {gap:.2f} dB | "
                f"LR: {lr:.2e}"
            )

        # Best model & early stopping
        if save_best and va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": va_psnr,
                    "val_total_loss": va_total,
                    "val_data_loss": va_data,
                    "val_physics_loss": va_phys,
                    "physics_weight": physics_weight,
                    "wave_speed": wave_speed,
                    "grid_cols": grid_cols,
                    "grid_rows": grid_rows,
                    "patch_size": patch_size,
                    "dx": dx,
                    "dy": dy,
                    "model_type": model_type,
                    "base_channels": base_channels,
                    "coord_dim": coord_dim,
                },
                ckpt_path / "best_deepsets_pinn.pth",
            )
            tqdm.write(f"  → Saved best DeepSets PINN (Val PSNR: {va_psnr:.2f} dB)")
        else:
            early_stop_counter += 1
            if epoch >= min_epochs and early_stop_counter >= early_stopping_patience:
                tqdm.write(
                    f"\n[INFO] Early stopping at epoch {epoch}! "
                    f"No improvement for {early_stopping_patience} epochs."
                )
                break

    # ============================================================
    # Load Best Model
    # ============================================================
    best_path = ckpt_path / "best_deepsets_pinn.pth"
    if save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"\n[INFO] Loaded best model from epoch {ckpt['epoch']} "
            f"(Val PSNR: {ckpt['val_psnr']:.2f} dB)"
        )

    # ============================================================
    # Post-Training Visualisations
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] Generating post-training visualisations...")
    print("=" * 60)

    plot_training_curves(history, run_image_path("fig_deepsets_pinn_training_curves.png"))
    plot_sample_results(
        model, val_loader, device, run_image_path("fig_deepsets_pinn_results.png")
    )

    print("\n" + "=" * 60)
    print("[INFO] Running acoustic feature validation...")
    print("=" * 60)
    run_deepsets_acoustic_validation(
        model,
        val_loader,
        device,
        save_path=run_image_path("fig_deepsets_pinn_acoustic_validation.png"),
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("[INFO] DeepSets PINN Training Complete!")
    print("=" * 60)
    print(f"  → Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"  → Final training PSNR: {history['train_psnr'][-1]:.2f} dB")
    print(f"  → Final physics loss: {history['train_physics_loss'][-1]:.2e}")
    print(f"  → Physics weight (λ): {physics_weight}")
    print("  → Figures:")
    print(f"      - {run_image_path('fig_deepsets_pinn_pre_train_samples.png')}")
    print(f"      - {run_image_path('fig_deepsets_pinn_training_curves.png')}")
    print(f"      - {run_image_path('fig_deepsets_pinn_results.png')}")
    print(f"      - {run_image_path('fig_deepsets_pinn_acoustic_validation.png')}")
    print(f"  → Checkpoint: {best_path}")

    return model, history


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DeepSets PINN for Ultrasonic Signal Denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, default="data", help="Root data directory"
    )
    parser.add_argument("--grid_cols", type=int, default=41)
    parser.add_argument("--grid_rows", type=int, default=41)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=5,
        help="Patch side length (e.g. 5 → 25 elements/set)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Patch extraction stride")
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation (enabled by default)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture
    parser.add_argument(
        "--model_type",
        type=str,
        default="spatial_cae",
        choices=["spatial_cae", "deepsets"],
        help="Model type: spatial_cae (DeepCAE backbone with spatial aux) "
        "or deepsets (legacy DeepSetsPINN)",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=16,
        help="Base conv channel count (SpatialCAE default=16, Deepsets default=16)",
    )
    parser.add_argument(
        "--coord_dim",
        type=int,
        default=64,
        help="Coordinate embedding dim for SpatialCAE model",
    )

    # Physics
    parser.add_argument(
        "--physics_weight",
        type=float,
        default=0.0001,
        help="Weight λ for wave equation physics loss",
    )
    parser.add_argument(
        "--wave_speed", type=float, default=5900.0, help="Speed of sound (m/s)"
    )
    parser.add_argument(
        "--center_frequency",
        type=float,
        default=250e3,
        help="Transducer centre frequency (Hz)",
    )
    parser.add_argument(
        "--dx", type=float, default=GRID_SPACING, help="Grid x-spacing (m)"
    )
    parser.add_argument(
        "--dy", type=float, default=GRID_SPACING, help="Grid y-spacing (m)"
    )

    args = parser.parse_args()

    model, history = train_deepsets_pinn(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_path=args.data_path,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        patch_size=args.patch_size,
        stride=args.stride,
        dropout_rate=args.dropout,
        augment=not args.no_augment,
        model_type=args.model_type,
        base_channels=args.base_channels,
        coord_dim=args.coord_dim,
        physics_weight=args.physics_weight,
        wave_speed=args.wave_speed,
        center_frequency=args.center_frequency,
        dx=args.dx,
        dy=args.dy,
        early_stopping_patience=args.patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
    )
