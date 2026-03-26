"""Unified training entry for PINN and DeepSets-PINN pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data import GRID_SPACING, create_dataloaders, create_deepsets_dataloaders
from model.model import DeepCAE_PINN, DeepSetsPINN, count_parameters
from scripts.analysis import acoustic_validation as av
from scripts.analysis.acoustic_validation import run_acoustic_validation
from scripts.train.visualization import plot_results


RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_image_dir(checkpoint_dir: str) -> Path:
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.name == "checkpoints":
        return ckpt_path.parent / "images"
    return RESULTS_DIR / "images"


def calculate_psnr(
    clean: torch.Tensor, denoised: torch.Tensor, max_val: float = 1.0
) -> float:
    mse = torch.mean((clean - denoised) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def calculate_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    signal_power = torch.mean(signal**2).item()
    noise_power = torch.mean(noise**2).item()
    if noise_power < 1e-10:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def plot_pre_training_samples(
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    num_samples: int = 3,
) -> None:
    noisy, clean = next(iter(dataloader))
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
        noisy_signal = noisy[i, 0].cpu().numpy()
        clean_signal = clean[i, 0].cpu().numpy()
        noise = noisy_signal - clean_signal
        snr = calculate_snr(torch.from_numpy(clean_signal), torch.from_numpy(noise))

        axes[i, 0].plot(time_us, noisy_signal, "b-", linewidth=0.7, alpha=0.8)
        axes[i, 0].set_title(f"Sample {i + 1}: Noisy Input (SNR: {snr:.1f} dB)")
        axes[i, 0].set_xlabel("Time (us)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(time_us, clean_signal, "g-", linewidth=0.8)
        axes[i, 1].set_title(f"Sample {i + 1}: Clean Ground Truth")
        axes[i, 1].set_xlabel("Time (us)")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved pre-training samples to {save_path}")


def plot_pre_training_samples_deepsets(
    dataloader: torch.utils.data.DataLoader,
    save_path: str,
    num_samples: int = 3,
) -> None:
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
        center = noisy.shape[1] // 2
        noisy_signal = noisy[i, center].cpu().numpy()
        clean_signal = clean[i, center].cpu().numpy()
        noise = noisy_signal - clean_signal
        snr = calculate_snr(torch.from_numpy(clean_signal), torch.from_numpy(noise))

        axes[i, 0].plot(time_us, noisy_signal, "b-", linewidth=0.7, alpha=0.8)
        axes[i, 0].set_title(f"Sample {i + 1}: Noisy Input (SNR: {snr:.1f} dB)")
        axes[i, 0].set_xlabel("Time (us)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(time_us, clean_signal, "g-", linewidth=0.8)
        axes[i, 1].set_title(f"Sample {i + 1}: Clean Ground Truth")
        axes[i, 1].set_xlabel("Time (us)")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved pre-training samples to {save_path}")


class PINNLoss(nn.Module):
    def __init__(self, physics_weight: float = 0.001):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_weight = physics_weight

    def forward(
        self, denoised: torch.Tensor, clean: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_loss = self.data_loss_fn(denoised, clean)
        physics_loss = torch.mean(residual**2)
        total = data_loss + self.physics_weight * physics_loss
        return total, data_loss, physics_loss


class DeepSetsPINNLoss(nn.Module):
    def __init__(self, physics_weight: float = 0.001):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_weight = physics_weight

    def forward(
        self, denoised: torch.Tensor, clean: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_loss = self.data_loss_fn(denoised, clean)
        physics_loss = torch.mean(residual**2)
        total = data_loss + self.physics_weight * physics_loss
        return total, data_loss, physics_loss


def train_epoch_pinn(
    model: DeepCAE_PINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: PINNLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.train()
    total_sum, data_sum, phys_sum, psnr_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        denoised, residual = model.physics_forward(noisy)
        total, data_loss, phys_loss = criterion(denoised, clean, residual)

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_sum += total.item()
        data_sum += data_loss.item()
        phys_sum += phys_loss.item()
        psnr_sum += calculate_psnr(clean, denoised)
        n += 1

    return total_sum / n, data_sum / n, phys_sum / n, psnr_sum / n


def validate_pinn(
    model: DeepCAE_PINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: PINNLoss,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_sum, data_sum, phys_sum, psnr_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            denoised, residual = model.physics_forward(noisy)
            total, data_loss, phys_loss = criterion(denoised, clean, residual)
            total_sum += total.item()
            data_sum += data_loss.item()
            phys_sum += phys_loss.item()
            psnr_sum += calculate_psnr(clean, denoised)
            n += 1
    return total_sum / n, data_sum / n, phys_sum / n, psnr_sum / n


def train_epoch_deepsets(
    model: DeepSetsPINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: DeepSetsPINNLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    grid_cols: int,
    grid_rows: int,
) -> Tuple[float, float, float, float]:
    model.train()
    total_sum, data_sum, phys_sum, psnr_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in dataloader:
        noisy = batch["noisy_signals"].to(device)
        clean = batch["clean_signals"].to(device)
        coords = batch["coordinates"].to(device)
        grid_idx = batch["grid_indices"].to(device)

        denoised, residual = model.physics_forward(
            noisy, coords, grid_idx, grid_cols, grid_rows
        )
        total, data_loss, phys_loss = criterion(denoised, clean, residual)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_sum += total.item()
        data_sum += data_loss.item()
        phys_sum += phys_loss.item()
        psnr_sum += calculate_psnr(clean, denoised)
        n += 1

    return total_sum / n, data_sum / n, phys_sum / n, psnr_sum / n


def validate_deepsets(
    model: DeepSetsPINN,
    dataloader: torch.utils.data.DataLoader,
    criterion: DeepSetsPINNLoss,
    device: torch.device,
    grid_cols: int,
    grid_rows: int,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_sum, data_sum, phys_sum, psnr_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            noisy = batch["noisy_signals"].to(device)
            clean = batch["clean_signals"].to(device)
            coords = batch["coordinates"].to(device)
            grid_idx = batch["grid_indices"].to(device)
            denoised, residual = model.physics_forward(
                noisy, coords, grid_idx, grid_cols, grid_rows
            )
            total, data_loss, phys_loss = criterion(denoised, clean, residual)

            total_sum += total.item()
            data_sum += data_loss.item()
            phys_sum += phys_loss.item()
            psnr_sum += calculate_psnr(clean, denoised)
            n += 1
    return total_sum / n, data_sum / n, phys_sum / n, psnr_sum / n


def plot_pinn_training_curves(history: Dict[str, list], save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PINN Training Progress", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["train_total_loss"]) + 1)

    axes[0].plot(
        epochs, history["train_total_loss"], "b-", label="Train Total", linewidth=1.5
    )
    axes[0].plot(
        epochs, history["val_total_loss"], "r-", label="Val Total", linewidth=1.5
    )
    axes[0].set_title("Total Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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
    axes[1].set_title("Loss Decomposition")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_psnr"], "b-", label="Train PSNR", linewidth=1.5)
    axes[2].plot(epochs, history["val_psnr"], "r-", label="Val PSNR", linewidth=1.5)
    axes[2].set_title("PSNR Curve")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved training curves to {save_path}")


def plot_deepsets_sample_results(
    model: DeepSetsPINN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    n_samples: int = 6,
) -> None:
    model.eval()
    all_noisy, all_clean, all_coords = [], [], []
    for batch in dataloader:
        all_noisy.append(batch["noisy_signals"])
        all_clean.append(batch["clean_signals"])
        all_coords.append(batch["coordinates"])
        if len(all_noisy) * batch["noisy_signals"].shape[0] >= 100:
            break

    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    all_coords = torch.cat(all_coords, dim=0)
    total = all_noisy.shape[0]
    n_samples = min(n_samples, total)
    idx = np.random.choice(total, size=n_samples, replace=False)

    noisy = all_noisy[idx]
    clean = all_clean[idx]
    coords = all_coords[idx]

    with torch.no_grad():
        denoised = model(noisy.to(device), coords.to(device))

    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()
    denoised_np = denoised.cpu().numpy()

    time_us = np.linspace(0, 160, noisy_np.shape[-1])
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 10))
    if n_samples == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    row_titles = ["Noisy Input", "Clean Ground Truth", "Denoised Output"]
    colors = ["blue", "green", "red"]
    for col in range(n_samples):
        center = noisy_np[col].shape[0] // 2
        signals = [
            noisy_np[col, center],
            clean_np[col, center],
            denoised_np[col, center],
        ]
        psnr = calculate_psnr(
            torch.from_numpy(clean_np[col, center : center + 1]),
            torch.from_numpy(denoised_np[col, center : center + 1]),
        )
        input_noise = signals[0] - signals[1]
        input_snr = calculate_snr(
            torch.from_numpy(signals[1]), torch.from_numpy(input_noise)
        )

        for row in range(3):
            ax = axes[row, col]
            ax.plot(time_us, signals[row], color=colors[row], linewidth=0.7)
            if row == 0:
                ax.set_title(
                    f"Sample {col + 1}\\n{row_titles[row]}\\n(Input SNR: {input_snr:.1f} dB)",
                    fontsize=10,
                )
            elif row == 2:
                ax.set_title(f"{row_titles[row]}\\n(PSNR: {psnr:.2f} dB)", fontsize=10)
            else:
                ax.set_title(row_titles[row], fontsize=10)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved sample results to {save_path}")


def run_deepsets_acoustic_validation(
    model: DeepSetsPINN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 20,
) -> Dict[str, object]:
    model.eval()
    all_noisy, all_clean, all_coords = [], [], []
    for batch in dataloader:
        all_noisy.append(batch["noisy_signals"])
        all_clean.append(batch["clean_signals"])
        all_coords.append(batch["coordinates"])
        if sum(x.shape[0] for x in all_noisy) >= 100:
            break

    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    all_coords = torch.cat(all_coords, dim=0)
    n = min(num_samples, all_noisy.shape[0])
    indices = np.random.choice(all_noisy.shape[0], size=n, replace=False)

    noisy = all_noisy[indices]
    clean = all_clean[indices]
    coords = all_coords[indices]

    with torch.no_grad():
        predicted = model(noisy.to(device), coords.to(device))

    center = noisy.shape[1] // 2
    inp_np = noisy.cpu().numpy()[:, center, :]
    tgt_np = clean.cpu().numpy()[:, center, :]
    pred_np = predicted.cpu().numpy()[:, center, :]

    all_features = []
    xcorr_tp_list, env_corr_tp_list, coherence_tp_list = [], [], []
    arrival_errors, peak_pres_list, rms_pres_list = [], [], []
    dom_freq_pres_list, energy_match_list = [], []

    for i in range(n):
        all_features.append(
            {
                "input": av._extract_all_features(inp_np[i]),
                "target": av._extract_all_features(tgt_np[i]),
                "predicted": av._extract_all_features(pred_np[i]),
            }
        )

        peak_corr, _ = av._cross_correlation_peak(tgt_np[i], pred_np[i])
        xcorr_tp_list.append(peak_corr)
        env_corr_tp_list.append(av._envelope_correlation(tgt_np[i], pred_np[i]))

        freq_coh, coh = av._spectral_coherence(tgt_np[i], pred_np[i])
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
            [
                fa["predicted"].get(f"energy_ratio_{band}", 0)
                for band in av.SUB_BAND_LABELS
            ]
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
        all_features=all_features,
        quality_metrics=quality_metrics,
        save_path=save_path,
    )
    av._print_report(quality_metrics)
    return {"features": all_features, "quality_metrics": quality_metrics}


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    if torch.backends.mps.is_available():
        print("[INFO] Using Apple MPS (Metal Performance Shaders)")
        return torch.device("mps")
    print("[INFO] Using CPU")
    return torch.device("cpu")


def train_pinn(
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_train: int = 5000,
    num_val: int = 1000,
    save_best: bool = True,
    checkpoint_dir: str = str(CHECKPOINTS_DIR),
    seed: int = 42,
    data_mode: str = "synthetic",
    data_path: str | None = None,
    early_stopping_patience: int = 50,
    min_epochs: int = 30,
    dropout_rate: float = 0.1,
    augment: bool = False,
    physics_weight: float = 0.001,
    wave_speed: float = 5900.0,
    center_frequency: float = 250e3,
    damping_ratio: float = 0.05,
) -> Tuple[nn.Module, Dict[str, list]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _select_device()

    if data_mode == "file":
        print(f"\n[INFO] Loading experimental data from {data_path}...")
        train_loader, val_loader = create_dataloaders(
            batch_size=batch_size,
            seed=seed,
            mode="file",
            data_path=data_path,
            augment=augment,
        )
    else:
        print("\n[INFO] Creating synthetic datasets...")
        train_loader, val_loader = create_dataloaders(
            num_train=num_train,
            num_val=num_val,
            batch_size=batch_size,
            seed=seed,
            mode="synthetic",
            augment=augment,
        )

    model = DeepCAE_PINN(
        dropout_rate=dropout_rate,
        wave_speed=wave_speed,
        center_frequency=center_frequency,
        damping_ratio=damping_ratio,
    ).to(device)
    print(f"\n[INFO] Model: DeepCAE_PINN")
    print(f"[INFO] Total parameters: {count_parameters(model):,}")

    criterion = PINNLoss(physics_weight=physics_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6
    )

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    image_dir = _resolve_image_dir(checkpoint_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    run_image = lambda f: str(image_dir / f)

    plot_pre_training_samples(train_loader, run_image("fig_pinn_pre_train_samples.png"))

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
    early_counter = 0

    pbar = tqdm(range(1, num_epochs + 1), desc="PINN Training", unit="epoch")
    for epoch in pbar:
        tr_total, tr_data, tr_phys, tr_psnr = train_epoch_pinn(
            model, train_loader, criterion, optimizer, device
        )
        va_total, va_data, va_phys, va_psnr = validate_pinn(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)

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

        if save_best and va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            early_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": va_psnr,
                    "val_total_loss": va_total,
                    "val_data_loss": va_data,
                    "val_physics_loss": va_phys,
                    "train_psnr": tr_psnr,
                    "physics_weight": physics_weight,
                    "wave_speed": wave_speed,
                },
                checkpoint_path / "best_pinn_model.pth",
            )
        else:
            early_counter += 1
            if epoch >= min_epochs and early_counter >= early_stopping_patience:
                break

    best_ckpt = checkpoint_path / "best_pinn_model.pth"
    if save_best and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    plot_results(
        model,
        val_loader,
        device,
        run_image("fig_pinn_results.png"),
        train_config={
            "model": "DeepCAE_PINN",
            "epochs": num_epochs,
            "dropout": dropout_rate,
            "augment": augment,
            "mode": data_mode,
            "best_psnr": best_val_psnr,
        },
    )
    plot_pinn_training_curves(history, run_image("fig_pinn_training_curves.png"))
    run_acoustic_validation(
        model,
        val_loader,
        device,
        save_path=run_image("fig_pinn_acoustic_validation.png"),
    )
    return model, history


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
    grid_cols: int = 41,
    grid_rows: int = 41,
    patch_size: int = 5,
    stride: int = 1,
    physics_weight: float = 0.001,
    wave_speed: float = 5900.0,
    center_frequency: float = 250e3,
    dx: float = GRID_SPACING,
    dy: float = GRID_SPACING,
    model_type: str = "deepsets",
    base_channels: int = 16,
    coord_dim: int = 64,
    signal_embed_dim: int = 128,
    coord_embed_dim: int = 64,
    point_dim: int = 128,
) -> Tuple[nn.Module, Dict[str, list]]:
    del model_type, coord_dim
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _select_device()

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

    print(f"\n[INFO] Model: DeepSetsPINN")
    print(f"[INFO] Total parameters: {count_parameters(model):,}")

    criterion = DeepSetsPINNLoss(physics_weight=physics_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    image_dir = _resolve_image_dir(checkpoint_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    run_image = lambda f: str(image_dir / f)

    plot_pre_training_samples_deepsets(
        train_loader, run_image("fig_deepsets_pinn_pre_train_samples.png")
    )

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
    early_counter = 0
    pbar = tqdm(range(1, num_epochs + 1), desc="DeepSets PINN Training", unit="epoch")
    for epoch in pbar:
        tr_total, tr_data, tr_phys, tr_psnr = train_epoch_deepsets(
            model, train_loader, criterion, optimizer, device, grid_cols, grid_rows
        )
        va_total, va_data, va_phys, va_psnr = validate_deepsets(
            model, val_loader, criterion, device, grid_cols, grid_rows
        )
        scheduler.step(epoch)

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

        if save_best and va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            early_counter = 0
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
                    "model_type": "deepsets",
                    "base_channels": base_channels,
                    "coord_dim": coord_embed_dim,
                },
                ckpt_path / "best_deepsets_pinn.pth",
            )
        else:
            early_counter += 1
            if epoch >= min_epochs and early_counter >= early_stopping_patience:
                break

    best_path = ckpt_path / "best_deepsets_pinn.pth"
    if save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    plot_pinn_training_curves(
        history, run_image("fig_deepsets_pinn_training_curves.png")
    )
    plot_deepsets_sample_results(
        model, val_loader, device, run_image("fig_deepsets_pinn_results.png")
    )
    run_deepsets_acoustic_validation(
        model,
        val_loader,
        device,
        run_image("fig_deepsets_pinn_acoustic_validation.png"),
    )
    return model, history


def train_from_config(config: Dict[str, object]) -> Tuple[nn.Module, Dict[str, list]]:
    pipeline = str(config.get("pipeline", "pinn")).strip().lower()
    if pipeline == "pinn":
        return train_pinn(
            num_epochs=int(config.get("epochs", 50)),
            batch_size=int(config.get("batch_size", 32)),
            learning_rate=float(config.get("lr", 1e-3)),
            num_train=int(config.get("num_train", 5000)),
            num_val=int(config.get("num_val", 1000)),
            save_best=bool(config.get("save_best", True)),
            checkpoint_dir=str(config.get("checkpoint_dir", str(CHECKPOINTS_DIR))),
            seed=int(config.get("seed", 42)),
            data_mode=str(config.get("mode", config.get("data_mode", "synthetic"))),
            data_path=str(config.get("data_path", "data")),
            early_stopping_patience=int(config.get("patience", 50)),
            min_epochs=int(config.get("min_epochs", 30)),
            dropout_rate=float(config.get("dropout", 0.1)),
            augment=bool(config.get("augment", False)),
            physics_weight=float(config.get("physics_weight", 1e-3)),
            wave_speed=float(config.get("wave_speed", 5900.0)),
            center_frequency=float(config.get("center_frequency", 250e3)),
            damping_ratio=float(config.get("damping_ratio", 0.05)),
        )

    if pipeline == "deepsets":
        return train_deepsets_pinn(
            num_epochs=int(config.get("epochs", 50)),
            batch_size=int(config.get("batch_size", 32)),
            learning_rate=float(config.get("lr", 1e-3)),
            save_best=bool(config.get("save_best", True)),
            checkpoint_dir=str(config.get("checkpoint_dir", str(CHECKPOINTS_DIR))),
            seed=int(config.get("seed", 42)),
            data_path=str(config.get("data_path", "data")),
            early_stopping_patience=int(config.get("patience", 50)),
            min_epochs=int(config.get("min_epochs", 30)),
            dropout_rate=float(config.get("dropout", 0.1)),
            augment=bool(config.get("augment", True)),
            grid_cols=int(config.get("grid_cols", config.get("target_cols", 41))),
            grid_rows=int(config.get("grid_rows", config.get("target_rows", 41))),
            patch_size=int(config.get("patch_size", 5)),
            stride=int(config.get("stride", 1)),
            physics_weight=float(config.get("physics_weight", 1e-4)),
            wave_speed=float(config.get("wave_speed", 5900.0)),
            center_frequency=float(config.get("center_frequency", 250e3)),
            dx=float(config.get("dx", GRID_SPACING)),
            dy=float(config.get("dy", GRID_SPACING)),
            model_type="deepsets",
            base_channels=int(config.get("base_channels", 16)),
            coord_dim=int(config.get("coord_dim", 64)),
            signal_embed_dim=int(config.get("signal_embed_dim", 128)),
            coord_embed_dim=int(
                config.get("coord_embed_dim", config.get("coord_dim", 64))
            ),
            point_dim=int(config.get("point_dim", 128)),
        )

    raise ValueError(
        f"Unsupported pipeline: {pipeline}. Expected 'pinn' or 'deepsets'."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified trainer for PINN and DeepSets PINN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pipeline", choices=["pinn", "deepsets"], default="pinn")
    parser.add_argument(
        "--mode", type=str, default="synthetic", choices=["synthetic", "file"]
    )
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics_weight", type=float, default=None)
    parser.add_argument("--wave_speed", type=float, default=5900.0)
    parser.add_argument("--center_frequency", type=float, default=250e3)
    parser.add_argument("--damping_ratio", type=float, default=0.05)
    parser.add_argument("--grid_cols", type=int, default=41)
    parser.add_argument("--grid_rows", type=int, default=41)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--coord_dim", type=int, default=64)
    parser.add_argument("--signal_embed_dim", type=int, default=128)
    parser.add_argument("--coord_embed_dim", type=int, default=64)
    parser.add_argument("--point_dim", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    physics_weight = args.physics_weight
    if physics_weight is None:
        physics_weight = 1e-3 if args.pipeline == "pinn" else 1e-4

    config: Dict[str, object] = {
        "pipeline": args.pipeline,
        "mode": args.mode,
        "data_mode": args.mode,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "dropout": args.dropout,
        "augment": args.augment,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "seed": args.seed,
        "physics_weight": physics_weight,
        "wave_speed": args.wave_speed,
        "center_frequency": args.center_frequency,
        "damping_ratio": args.damping_ratio,
        "grid_cols": args.grid_cols,
        "grid_rows": args.grid_rows,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "base_channels": args.base_channels,
        "coord_dim": args.coord_dim,
        "signal_embed_dim": args.signal_embed_dim,
        "coord_embed_dim": args.coord_embed_dim,
        "point_dim": args.point_dim,
        "checkpoint_dir": args.checkpoint_dir,
        "save_best": True,
    }
    train_from_config(config)


if __name__ == "__main__":
    main()
