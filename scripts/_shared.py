"""Shared utilities for scripts/ — path helpers, metrics, and device selection."""

from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def image_path(filename: str) -> str:
    return str(IMAGES_DIR / filename)


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


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
