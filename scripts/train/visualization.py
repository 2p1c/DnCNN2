"""Visualization utilities for training result inspection."""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


from scripts._shared import calculate_psnr, calculate_snr, ensure_parent_dir


def plot_results(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 6,
    train_config: Optional[Dict[str, Any]] = None,
    unsupervised: bool = False,
) -> None:
    """
    Plot denoising/reconstruction results after training.

    Shows 3 rows (Noisy, Clean Reference, Output) x num_samples columns.

    Args:
        model: Trained model
        dataloader: DataLoader for test samples
        device: Device to run inference on
        save_path: Path to save the figure
        num_samples: Number of samples to visualize
        train_config: Training configuration for title display
        unsupervised: If True, display reconstruction-focused labels/metrics
    """
    model.eval()

    all_noisy = []
    all_clean = []
    for batch_noisy, batch_clean in dataloader:
        all_noisy.append(batch_noisy)
        all_clean.append(batch_clean)
        if len(all_noisy) * batch_noisy.shape[0] >= 100:
            break

    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)

    total_samples = all_noisy.shape[0]
    random_indices = np.random.choice(
        total_samples, size=min(num_samples, total_samples), replace=False
    )

    noisy = all_noisy[random_indices]
    clean = all_clean[random_indices]
    noisy_device = noisy.to(device)

    with torch.no_grad():
        denoised = model(noisy_device)

    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()
    denoised_np = denoised.cpu().numpy()

    time_us = np.linspace(0, 160, noisy_np.shape[-1])
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 10))

    if unsupervised:
        title = (
            "Unsupervised Reconstruction Results: "
            "Noisy Input → Clean Reference → Reconstructed Output"
        )
    else:
        title = "Denoising Results: Noisy Input → Clean Ground Truth → Denoised Output"

    if train_config:
        config_str = (
            f"model={train_config.get('model', 'N/A')}, "
            f"epochs={train_config.get('epochs', 'N/A')}, "
            f"dropout={train_config.get('dropout', 'N/A')}, "
            f"augment={train_config.get('augment', False)}, "
            f"mode={train_config.get('mode', 'N/A')}"
        )
        if train_config.get("best_psnr"):
            config_str += f", best_val_psnr={train_config['best_psnr']:.2f}dB"
        title = f"{title}\n[{config_str}]"

    fig.suptitle(title, fontsize=12, fontweight="bold")

    if unsupervised:
        row_titles = [
            "Noisy Input",
            "Clean Reference (not training target)",
            "Reconstructed Output",
        ]
    else:
        row_titles = ["Noisy Input", "Clean Ground Truth", "Denoised Output"]

    colors = ["blue", "green", "red"]

    for col in range(num_samples):
        noisy_sig = noisy_np[col, 0]
        clean_sig = clean_np[col, 0]
        denoised_sig = denoised_np[col, 0]
        signals = [noisy_sig, clean_sig, denoised_sig]

        psnr_clean = calculate_psnr(
            torch.from_numpy(clean_np[col]), torch.from_numpy(denoised_np[col])
        )
        psnr_noisy = calculate_psnr(
            torch.from_numpy(noisy_np[col]), torch.from_numpy(denoised_np[col])
        )

        noise = noisy_sig - clean_sig
        input_snr = calculate_snr(torch.from_numpy(clean_sig), torch.from_numpy(noise))

        for row in range(3):
            ax = axes[row, col] if num_samples > 1 else axes[row]
            ax.plot(time_us, signals[row], color=colors[row], linewidth=0.7)

            if row == 0:
                ax.set_title(
                    f"Sample {col + 1}\n{row_titles[row]}\n(Input SNR: {input_snr:.1f} dB)",
                    fontsize=10,
                )
            elif row == 2:
                if unsupervised:
                    ax.set_title(
                        f"{row_titles[row]}\n"
                        f"(PSNR vs Noisy target: {psnr_noisy:.2f} dB; "
                        f"vs Clean ref: {psnr_clean:.2f} dB)",
                        fontsize=9,
                    )
                else:
                    ax.set_title(
                        f"{row_titles[row]}\n(PSNR: {psnr_clean:.2f} dB)",
                        fontsize=10,
                    )
            else:
                ax.set_title(row_titles[row], fontsize=10)

            ax.set_xlabel("Time (μs)")
            ax.set_ylabel("Amplitude")
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved results to {save_path}")


def plot_inference_comparison(
    original_signals: np.ndarray,
    inferred_signals: np.ndarray,
    save_path: str,
    num_samples: int = 6,
    random_seed: int = 42,
) -> None:
    """
    Plot inference comparison results.

    Shows 2 rows (Original Signal, Inference Result) x num_samples columns.

    Args:
        original_signals: Original signals, shape (N, L) or compatible.
        inferred_signals: Inference outputs, shape (N, L) or compatible.
        save_path: Path to save the figure.
        num_samples: Number of samples to visualize.
        random_seed: Random seed for reproducible sample selection.
    """
    original_arr = np.asarray(original_signals)
    inferred_arr = np.asarray(inferred_signals)

    if original_arr.size == 0 or inferred_arr.size == 0:
        raise ValueError("No signals available for plotting")

    # Accept both (N, L) and (N, 1, L) inputs
    if original_arr.ndim == 3 and original_arr.shape[1] == 1:
        original_arr = original_arr[:, 0, :]
    if inferred_arr.ndim == 3 and inferred_arr.shape[1] == 1:
        inferred_arr = inferred_arr[:, 0, :]

    if original_arr.ndim != 2 or inferred_arr.ndim != 2:
        raise ValueError(
            "Signals must be 2D arrays with shape (N, L) or 3D with shape (N, 1, L)"
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    total_samples = min(original_arr.shape[0], inferred_arr.shape[0])
    sample_count = min(num_samples, total_samples)

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(total_samples, size=sample_count, replace=False)
    print(
        f"[INFO] Inference figure sampled {sample_count} pairs "
        f"with random_seed={random_seed}: {sample_indices.tolist()}"
    )

    time_us = np.linspace(0, 160, original_arr.shape[-1])
    fig, axes = plt.subplots(2, sample_count, figsize=(4 * sample_count, 6))

    if sample_count == 1:
        axes = np.expand_dims(axes, axis=1)

    fig.suptitle(
        "Inference Results: Original Signal → Inference Output",
        fontsize=12,
        fontweight="bold",
    )

    for col, idx in enumerate(sample_indices):
        original_sig = original_arr[idx].astype(np.float64, copy=False)
        inferred_sig = inferred_arr[idx].astype(np.float64, copy=False)

        # Visualization-only normalization to avoid near-zero flat lines.
        # Use the same scale for each pair to keep before/after comparable.
        pair_peak = max(float(np.max(np.abs(original_sig))), float(np.max(np.abs(inferred_sig))))
        if pair_peak > 1e-12:
            original_plot = original_sig / pair_peak
            inferred_plot = inferred_sig / pair_peak
            title_suffix = "(normalized)"
        else:
            original_plot = np.zeros_like(original_sig)
            inferred_plot = np.zeros_like(inferred_sig)
            title_suffix = "(near-constant)"

        axes[0, col].plot(time_us, original_plot, color="blue", linewidth=0.7)
        axes[0, col].set_title(
            f"Sample {idx + 1}\nOriginal Signal {title_suffix}", fontsize=10
        )
        axes[0, col].set_xlabel("Time (μs)")
        axes[0, col].set_ylabel("Amplitude")
        axes[0, col].set_ylim(-1.1, 1.1)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

        axes[1, col].plot(time_us, inferred_plot, color="red", linewidth=0.7)
        axes[1, col].set_title(
            f"Sample {idx + 1}\nInference Result {title_suffix}", fontsize=10
        )
        axes[1, col].set_xlabel("Time (μs)")
        axes[1, col].set_ylabel("Amplitude")
        axes[1, col].set_ylim(-1.1, 1.1)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axhline(y=0, color="k", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Saved inference comparison figure to {save_path}")
