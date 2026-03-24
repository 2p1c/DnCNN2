"""
Inference Script for DeepSets PINN Ultrasonic Signal Denoising

Loads a trained DeepSets PINN model and denoises experimental data.
The script reconstructs the full grid from patches, averages overlapping
predictions, denormalises signals, and saves results to .mat format.

Usage:
    uv run python inference_deepsets.py --input noisy.mat --output results/
    uv run python inference_deepsets.py --input noisy.mat --checkpoint results/checkpoints/best_deepsets_pinn.pth
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules like model, data_utils can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import io as sio
from pathlib import Path
from datetime import datetime
from typing import Tuple

from model.model import (
    SetInvariantWavePINN,
    SpatialContextCAE,
)
from data import GRID_SPACING
from scripts.transformer import (
    load_mat_file,
    reshape_to_grid,
    flatten_grid,
    interpolate_spatial,
    normalize_signal,
    truncate_signals,
)
from scripts.analysis.acoustic_validation import run_inference_validation


RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"


# ============================================================
# Model Loading
# ============================================================


def load_model(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[nn.Module, dict]:
    """
    Load trained DeepSets PINN from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint
        device: Target device

    Returns:
        (model, checkpoint_metadata)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[INFO] Loading model from {checkpoint_path}")
    print(f"[INFO] Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model with saved config
    dx = ckpt.get("dx", GRID_SPACING)
    dy = ckpt.get("dy", GRID_SPACING)
    wave_speed = ckpt.get("wave_speed", 5900.0)
    model_type = str(ckpt.get("model_type", "deepsets")).strip().lower()

    if model_type in {"spatial_cae", "spatial_context_cae"}:
        base_channels = ckpt.get("base_channels", 32)
        coord_dim = ckpt.get("coord_dim", 64)
        model = SpatialContextCAE(
            base_channels=base_channels,
            coord_dim=coord_dim,
            dropout_rate=0.0,
            wave_speed=wave_speed,
            dx=dx,
            dy=dy,
        )
    else:
        model = SetInvariantWavePINN(
            dropout_rate=0.0,  # No dropout at inference
            wave_speed=wave_speed,
            dx=dx,
            dy=dy,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"[INFO] Model loaded: {model.__class__.__name__} "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"val PSNR {ckpt.get('val_psnr', '?'):.2f} dB)"
    )
    return model, ckpt


# ============================================================
# Preprocessing
# ============================================================


def preprocess_mat_data(
    mat_path: str,
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    interpolation_method: str = "cubic",
    target_signal_length: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess .mat file: load, interpolate, normalise.

    Returns:
        (normalised_signals, original_signals, metadata)
        Both arrays have shape (n_signals, signal_length).
    """
    print(f"\n[Step 1] Loading .mat file: {mat_path}")
    time_vector, signal_data = load_mat_file(mat_path)
    time_vector, signal_data = truncate_signals(
        time_vector, signal_data, target_signal_length
    )

    metadata = {
        "time_vector": time_vector,
        "signal_length": len(time_vector),
        "original_shape": signal_data.shape,
        "input_grid_size": (grid_cols, grid_rows),
        "output_grid_size": (target_cols, target_rows),
    }

    n_input = grid_cols * grid_rows
    n_target = target_cols * target_rows

    if signal_data.shape[0] == n_target:
        print(f"[INFO] Already at target size, skipping interpolation")
        processed = signal_data.copy()
        metadata["interpolated"] = False
    elif signal_data.shape[0] == n_input:
        print(f"\n[Step 2] Reshaping to {grid_cols}×{grid_rows} grid...")
        grid = reshape_to_grid(signal_data, grid_cols, grid_rows)
        print(f"\n[Step 3] Interpolating to {target_cols}×{target_rows}...")
        grid_interp = interpolate_spatial(
            grid, target_cols, target_rows, method=interpolation_method
        )
        print(f"\n[Step 4] Flattening grid...")
        processed = flatten_grid(grid_interp)
        metadata["interpolated"] = True
    else:
        raise ValueError(
            f"Unexpected point count {signal_data.shape[0]}. "
            f"Expected {n_input} or {n_target}."
        )

    original = processed.copy()

    # Per-signal normalisation
    signal_mins, signal_maxs = [], []
    normalised = np.zeros_like(processed, dtype=np.float32)
    for i in range(processed.shape[0]):
        sig_min, sig_max = processed[i].min(), processed[i].max()
        signal_mins.append(sig_min)
        signal_maxs.append(sig_max)
        normalised[i] = normalize_signal(processed[i])

    metadata["signal_mins"] = np.array(signal_mins)
    metadata["signal_maxs"] = np.array(signal_maxs)

    print(f"[INFO] Preprocessed {normalised.shape[0]} signals")
    return normalised, original, metadata


# ============================================================
# Grid Denoising
# ============================================================


def denoise_grid(
    model: nn.Module,
    normalised_signals: np.ndarray,
    grid_cols: int = 41,
    grid_rows: int = 41,
    patch_size: int = 5,
    device: torch.device = None,
) -> np.ndarray:
    """
    Denoise the full grid by sweeping patches across it.

    For each interior centre point, extract a patch_size×patch_size
    neighbourhood, run the model, and accumulate the centre signal
    prediction.

    Args:
        model: Trained denoising model
        normalised_signals: (n_signals, T) normalised input
        grid_cols, grid_rows: Grid dimensions
        patch_size: Patch side length
        device: Compute device

    Returns:
        denoised: (n_signals, T) normalised denoised signals
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_signals = grid_cols * grid_rows
    T = normalised_signals.shape[1]
    half = patch_size // 2

    # Accumulators for averaging overlapping predictions
    denoised_sum = np.zeros((n_signals, T), dtype=np.float64)
    count = np.zeros(n_signals, dtype=np.float64)

    # Build coordinate grid (same as DeepSetsDataset)
    cols_arr = np.arange(grid_cols)
    rows_arr = np.arange(grid_rows)
    cc, rr = np.meshgrid(cols_arr, rows_arr, indexing="ij")
    coords_all = np.stack(
        [cc.flatten() / max(grid_cols - 1, 1), rr.flatten() / max(grid_rows - 1, 1)],
        axis=-1,
    ).astype(np.float32)

    total_patches = (grid_cols - 2 * half) * (grid_rows - 2 * half)
    print(f"\n[INFO] Denoising {total_patches} patches ({patch_size}×{patch_size})...")

    processed = 0
    with torch.no_grad():
        for ci in range(half, grid_cols - half):
            for cj in range(half, grid_rows - half):
                # Extract patch indices
                indices = []
                for dc in range(-half, half + 1):
                    for dr in range(-half, half + 1):
                        indices.append((ci + dc) * grid_rows + (cj + dr))
                indices = np.array(indices)

                # Prepare tensors (batch_size=1)
                sig = (
                    torch.from_numpy(normalised_signals[indices])
                    .unsqueeze(0)
                    .to(device)
                )
                coord = torch.from_numpy(coords_all[indices]).unsqueeze(0).to(device)

                # Forward
                out = model(sig, coord)  # (1, R, T)
                out_np = out.squeeze(0).cpu().numpy()

                # Accumulate all R predictions
                for k, flat_idx in enumerate(indices):
                    denoised_sum[flat_idx] += out_np[k]
                    count[flat_idx] += 1

                processed += 1
                if processed % 200 == 0 or processed == total_patches:
                    print(f"  Processed {processed}/{total_patches} patches")

    # Average overlapping predictions
    mask = count > 0
    denoised_sum[mask] /= count[mask, np.newaxis]

    # Edge points with no prediction: copy input
    no_pred = ~mask
    if no_pred.any():
        print(
            f"  [WARN] {no_pred.sum()} edge points had no prediction, "
            f"using input signal"
        )
        denoised_sum[no_pred] = normalised_signals[no_pred]

    return denoised_sum.astype(np.float32)


# ============================================================
# Denormalisation
# ============================================================


def denormalize_signals(
    denoised_normalised: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """Reverse normalisation to original amplitude scale."""
    mins = metadata["signal_mins"]
    maxs = metadata["signal_maxs"]
    denoised = np.zeros_like(denoised_normalised)

    for i in range(denoised_normalised.shape[0]):
        amp = maxs[i] - mins[i]
        if amp > np.finfo(np.float64).eps * 0.1:
            denoised[i] = (denoised_normalised[i] + 1) / 2 * amp + mins[i]
        else:
            denoised[i] = np.full_like(denoised_normalised[i], mins[i])

    return denoised


# ============================================================
# Main Inference Pipeline
# ============================================================


def run_inference(
    input_path: str,
    output_path: str,
    checkpoint_path: str = str(CHECKPOINTS_DIR / "best_deepsets_pinn.pth"),
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    patch_size: int = 5,
    interpolation_method: str = "cubic",
    target_signal_length: int = 1000,
    validation_save_path: str | None = None,
) -> str:
    """
    Full inference pipeline for DeepSets PINN.

    Args:
        input_path: Input .mat file
        output_path: Output directory
        checkpoint_path: Model checkpoint
        grid_cols/rows: Input grid dimensions
        target_cols/rows: Interpolated grid dimensions
        patch_size: Patch side length
        interpolation_method: Spatial interpolation method
        target_signal_length: Signal truncation length

    Returns:
        Path to saved output .mat file
    """
    print("=" * 60)
    print("DeepSets PINN — Ultrasonic Signal Inference")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model, ckpt = load_model(checkpoint_path, device)

    # Preprocess
    normalised, original, metadata = preprocess_mat_data(
        input_path,
        grid_cols,
        grid_rows,
        target_cols,
        target_rows,
        interpolation_method,
        target_signal_length,
    )

    # Denoise full grid
    denoised_norm = denoise_grid(
        model, normalised, target_cols, target_rows, patch_size, device
    )

    # Denormalise
    print("\n[INFO] Denormalising signals...")
    denoised = denormalize_signals(denoised_norm, metadata)

    # Save
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("[INFO] Running acoustic feature validation...")
    print("=" * 60)
    if validation_save_path:
        validation_fig = validation_save_path
    else:
        validation_fig = str(image_dir / f"acoustic_validation_deepsets_{timestamp}.png")
    run_inference_validation(
        input_signals=normalised,
        denoised_signals=denoised_norm,
        save_path=validation_fig,
    )

    out_file = out_dir / f"deepsets_denoised_{timestamp}.mat"
    sio.savemat(
        str(out_file),
        {
            "x": metadata["time_vector"],
            "y": denoised,
            "signal_length": metadata["signal_length"],
            "grid_cols": target_cols,
            "grid_rows": target_rows,
            "n_signals": denoised.shape[0],
        },
    )
    print(f"\n[INFO] Saved denoised data to: {out_file}")

    # Optionally reverse interpolation
    if metadata.get("interpolated", False):
        from scripts.transformer import interpolate_spatial as interp_sp

        grid_big = reshape_to_grid(denoised, target_cols, target_rows)
        grid_small = interp_sp(
            grid_big, grid_cols, grid_rows, method=interpolation_method
        )
        denoised_small = flatten_grid(grid_small)

        out_file_orig = out_dir / f"deepsets_denoised_{timestamp}_original.mat"
        sio.savemat(
            str(out_file_orig),
            {
                "x": metadata["time_vector"],
                "y": denoised_small,
                "signal_length": metadata["signal_length"],
                "grid_cols": grid_cols,
                "grid_rows": grid_rows,
                "n_signals": denoised_small.shape[0],
            },
        )
        print(f"[INFO] Saved original-size data to: {out_file_orig}")

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"  Acoustic validation: {validation_fig}")
    return str(out_file)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSets PINN Inference for Ultrasonic Denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input .mat file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=str(RESULTS_DIR), help="Output directory"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_deepsets_pinn.pth"),
        help="Model checkpoint",
    )
    parser.add_argument("--cols", type=int, default=21)
    parser.add_argument("--rows", type=int, default=21)
    parser.add_argument("--target_cols", type=int, default=41)
    parser.add_argument("--target_rows", type=int, default=41)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument(
        "--interp_method", type=str, default="cubic", choices=["linear", "cubic"]
    )
    parser.add_argument("--signal_length", type=int, default=1000)

    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        grid_cols=args.cols,
        grid_rows=args.rows,
        target_cols=args.target_cols,
        target_rows=args.target_rows,
        patch_size=args.patch_size,
        interpolation_method=args.interp_method,
        target_signal_length=args.signal_length,
    )
