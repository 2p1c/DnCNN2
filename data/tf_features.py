"""Utilities for building 1D time-frequency features from ultrasonic signals."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from scipy.signal import stft


PoolingMode = Literal["mean", "max", "meanmax"]


def normalize_to_minus1_1(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize a 1D signal to [-1, 1] and return (normalized, min, max)."""
    sig = signal.astype(np.float64)
    min_val = float(sig.min())
    max_val = float(sig.max())
    amp = max_val - min_val
    if amp > np.finfo(np.float64).eps * 0.1:
        normalized = 2.0 * (sig - min_val) / amp - 1.0
    else:
        normalized = np.zeros_like(sig, dtype=np.float64)
    return normalized.astype(np.float32), min_val, max_val


def stft_to_1d_feature(
    signal_1d: np.ndarray,
    target_length: int,
    n_fft: int = 128,
    hop_length: int = 32,
    win_length: int = 128,
    window: str = "hann",
    pooling: PoolingMode = "mean",
) -> np.ndarray:
    """Convert one time-domain signal into a length-aligned 1D STFT feature."""
    if signal_1d.ndim != 1:
        raise ValueError(f"signal_1d must be 1D, got shape {signal_1d.shape}")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if hop_length <= 0 or win_length <= 0 or n_fft <= 0:
        raise ValueError("n_fft, hop_length and win_length must be positive")
    if win_length < hop_length:
        raise ValueError("win_length must be >= hop_length")
    if pooling not in {"mean", "max", "meanmax"}:
        raise ValueError(f"Unsupported pooling: {pooling}")

    _, _, zxx = stft(
        signal_1d.astype(np.float64),
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window=window,
        boundary=None,
        padded=False,
    )
    magnitude = np.log1p(np.abs(zxx))
    if pooling == "mean":
        feature = magnitude.mean(axis=0)
    elif pooling == "max":
        feature = magnitude.max(axis=0)
    else:
        feature = 0.5 * (magnitude.mean(axis=0) + magnitude.max(axis=0))

    x_old = np.linspace(0.0, 1.0, num=feature.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    return np.interp(x_new, x_old, feature).astype(np.float64)


def save_tf_feature(path: Path, feature: np.ndarray) -> None:
    """Save one tf feature as float32 .npy file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feature.astype(np.float32))


def build_and_save_tf_split(
    split_root: Path,
    signal_length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    pooling: PoolingMode,
) -> int:
    """Build tf features from `<split>/noisy/*.npy` into `<split>/tf/*.npy`."""
    noisy_dir = split_root / "noisy"
    tf_dir = split_root / "tf"
    if not noisy_dir.exists() or not noisy_dir.is_dir():
        raise FileNotFoundError(f"Missing noisy dir: {noisy_dir}")

    tf_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(noisy_dir.glob("*.npy"))
    for noisy_path in files:
        signal = np.load(noisy_path).astype(np.float64)
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal in {noisy_path}, got {signal.shape}")
        tf_raw = stft_to_1d_feature(
            signal_1d=signal,
            target_length=signal_length,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            pooling=pooling,
        )
        tf_norm, _, _ = normalize_to_minus1_1(tf_raw)
        save_tf_feature(tf_dir / noisy_path.name, tf_norm)
    return len(files)


def build_tf_features_from_processed_signals(
    processed_signals: np.ndarray,
    signal_length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    pooling: PoolingMode,
) -> np.ndarray:
    """Build normalized TF features for already-processed 2D signals array."""
    if processed_signals.ndim != 2:
        raise ValueError(
            f"processed_signals must be 2D (N,T), got {processed_signals.shape}"
        )
    tf = np.zeros_like(processed_signals, dtype=np.float32)
    for i in range(processed_signals.shape[0]):
        tf_raw = stft_to_1d_feature(
            signal_1d=processed_signals[i],
            target_length=signal_length,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            pooling=pooling,
        )
        tf_norm, _, _ = normalize_to_minus1_1(tf_raw)
        tf[i] = tf_norm
    return tf
