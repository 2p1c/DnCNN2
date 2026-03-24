"""
Inference Script for Ultrasonic Signal Denoising

Load trained model and denoise new experimental data from .mat files.
Converts input .mat data to model format, performs denoising, and
saves results back to .mat format.

Features:
- Load trained CAE model from checkpoint
- Process .mat files using transformer functions
- Batch inference on processed signals
- Save denoised results to .mat format
- Auto-naming results by date/time

Usage:
    # Basic usage with default model
    uv run python inference.py --input noisy.mat --output results/

    # With custom model checkpoint
    uv run python inference.py --input noisy.mat --output results/ --checkpoint results/checkpoints/best_model.pth

    # Specify grid size (if different from default 21x21)
    uv run python inference.py --input noisy.mat --output results/ --cols 21 --rows 21
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules like model, data_utils can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import numpy as np
import torch
from scipy import io as sio
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

from model.model import DeepCAE, DeeperCAE, LightweightCAE, UnsupervisedDeepCAE
from scripts.transformer import (
    load_mat_file,
    reshape_to_grid,
    flatten_grid,
    interpolate_spatial,
    normalize_signal,
    truncate_signals,
)
from scripts.analysis.acoustic_validation import run_inference_validation
from scripts.train.visualization import plot_inference_comparison


RESULTS_DIR = Path("results")
IMAGES_DIR = RESULTS_DIR / "images"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
TRAIN_SIGNAL_LENGTH = 1000


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for an output file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _print_signal_stats(tag: str, signals: np.ndarray) -> None:
    """Print concise signal statistics for debugging numerical issues."""
    if signals.size == 0:
        print(f"[WARNING] {tag}: empty signal array")
        return

    nonzero_ratio = float(np.count_nonzero(signals)) / float(signals.size)
    print(
        f"[INFO] {tag}: shape={signals.shape}, dtype={signals.dtype}, "
        f"min={signals.min():.6e}, max={signals.max():.6e}, "
        f"mean={signals.mean():.6e}, std={signals.std():.6e}, "
        f"nonzero_ratio={nonzero_ratio:.6f}"
    )


def load_model(
    checkpoint_path: str, model_type: str = "deep", device: torch.device = None
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        model_type: Model architecture ('lightweight', 'deeper', 'deep', 'unsupervised_deep')
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load checkpoint
    # Note: weights_only=False is needed for checkpoints saved with additional metadata
    # This is safe as long as the checkpoint comes from a trusted source
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format: state_dict is not a dictionary")

    enc1_weight = state_dict.get("enc1.0.weight")
    if enc1_weight is None:
        raise ValueError(
            "Checkpoint missing key 'enc1.0.weight'; cannot infer model channels"
        )

    base_channels = int(enc1_weight.shape[0])

    has_unsupervised_bottleneck = any(
        key.startswith("bottleneck_compress.") or key.startswith("bottleneck_expand.")
        for key in state_dict.keys()
    )
    checkpoint_unsupervised_flag = bool(checkpoint.get("unsupervised", False)) if isinstance(checkpoint, dict) else False

    resolved_model_type = model_type
    if has_unsupervised_bottleneck or checkpoint_unsupervised_flag:
        resolved_model_type = "unsupervised_deep"
        if model_type != "unsupervised_deep":
            print(
                "[WARNING] Checkpoint indicates unsupervised bottleneck architecture; "
                "overriding --model to 'unsupervised_deep'."
            )

    print(f"[INFO] Loading model from {checkpoint_path}")
    print(f"[INFO] Requested model type: {model_type}")
    print(f"[INFO] Resolved model type: {resolved_model_type}")
    print(f"[INFO] Inferred base_channels from checkpoint: {base_channels}")
    print(f"[INFO] Device: {device}")

    # Initialize model based on resolved type and inferred channels
    if resolved_model_type == "unsupervised_deep":
        bottleneck_weight = state_dict.get("bottleneck_compress.0.weight")
        if bottleneck_weight is None:
            raise ValueError(
                "Checkpoint indicates unsupervised model, but missing "
                "'bottleneck_compress.0.weight'"
            )
        bottleneck_channels = int(bottleneck_weight.shape[0])
        print(
            "[INFO] Inferred bottleneck_channels from checkpoint: "
            f"{bottleneck_channels}"
        )
        model = UnsupervisedDeepCAE(
            base_channels=base_channels,
            dropout_rate=0.0,
            bottleneck_channels=bottleneck_channels,
        )
    elif resolved_model_type == "deep":
        model = DeepCAE(base_channels=base_channels, dropout_rate=0.0)
    elif resolved_model_type == "deeper":
        model = DeeperCAE(base_channels=base_channels, dropout_rate=0.0)
    elif resolved_model_type == "lightweight":
        model = LightweightCAE(base_channels=base_channels, dropout_rate=0.0)
    else:
        raise ValueError(
            "Unsupported model_type: "
            f"{resolved_model_type}. Choose from lightweight/deeper/deep/unsupervised_deep"
        )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully!")
    return model


def preprocess_mat_data(
    mat_path: str,
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    interpolation_method: str = "cubic",
    target_signal_length: int = TRAIN_SIGNAL_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess .mat file data for inference.

    Loads the .mat file, reshapes to grid, interpolates spatially,
    and normalizes signals.

    Args:
        mat_path: Path to input .mat file
        grid_cols: Number of columns in input grid
        grid_rows: Number of rows in input grid
        target_cols: Target columns after interpolation
        target_rows: Target rows after interpolation
        interpolation_method: Interpolation method ('linear', 'cubic')
        target_signal_length: Target signal length (truncate if longer)

    Returns:
        Tuple of:
            - processed_signals: shape (n_signals, signal_length), normalized
            - original_signals: shape (n_signals, signal_length), unnormalized
            - metadata: dict with original data info for reconstruction
    """
    print(f"\n[Step 1] Loading .mat file: {mat_path}")
    time_vector, signal_data = load_mat_file(mat_path)

    # Truncate signals if longer than target length
    time_vector, signal_data = truncate_signals(
        time_vector, signal_data, target_signal_length
    )

    if signal_data.shape[1] != target_signal_length:
        raise ValueError(
            "Signal length mismatch after preprocessing: "
            f"got {signal_data.shape[1]}, expected {target_signal_length}. "
            "Please provide data with the same signal length as training."
        )

    # Store original data info for reconstruction
    metadata = {
        "time_vector": time_vector,
        "signal_length": len(time_vector),
        "original_shape": signal_data.shape,
        "input_grid_size": (grid_cols, grid_rows),
        "output_grid_size": (target_cols, target_rows),
        "original_min": signal_data.min(),
        "original_max": signal_data.max(),
    }

    # Check if interpolation is needed
    n_input_points = grid_cols * grid_rows
    n_target_points = target_cols * target_rows

    if signal_data.shape[0] == n_target_points:
        # Data is already at target size, no interpolation needed
        print(
            f"[INFO] Data already at target size ({target_cols}x{target_rows}), skipping interpolation"
        )
        processed_signals = signal_data.copy()
        metadata["interpolated"] = False
    elif signal_data.shape[0] == n_input_points:
        # Need to interpolate
        print(f"\n[Step 2] Reshaping to {grid_cols}x{grid_rows} grid...")
        input_grid = reshape_to_grid(signal_data, grid_cols, grid_rows)

        print(f"\n[Step 3] Interpolating to {target_cols}x{target_rows}...")
        output_grid = interpolate_spatial(
            input_grid, target_cols, target_rows, method=interpolation_method
        )

        print(f"\n[Step 4] Flattening grid...")
        processed_signals = flatten_grid(output_grid)
        metadata["interpolated"] = True
    else:
        raise ValueError(
            f"Unexpected number of data points: {signal_data.shape[0]}. "
            f"Expected {n_input_points} ({grid_cols}x{grid_rows}) or "
            f"{n_target_points} ({target_cols}x{target_rows})"
        )

    # Store unnormalized signals for later denormalization
    original_signals = processed_signals.copy()

    # Store per-signal normalization parameters
    signal_mins = []
    signal_maxs = []

    # Normalize each signal
    print(f"\n[Step 5] Normalizing {processed_signals.shape[0]} signals...")
    normalized_signals = np.zeros_like(processed_signals, dtype=np.float32)
    for i in range(processed_signals.shape[0]):
        sig = processed_signals[i]
        sig_min = sig.min()
        sig_max = sig.max()
        signal_mins.append(sig_min)
        signal_maxs.append(sig_max)
        normalized_signals[i] = normalize_signal(sig)

    metadata["signal_mins"] = np.array(signal_mins)
    metadata["signal_maxs"] = np.array(signal_maxs)

    print(f"[INFO] Preprocessing complete. Output shape: {normalized_signals.shape}")

    return normalized_signals, original_signals, metadata


def denoise_signals(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run inference on signals to denoise them.

    Args:
        model: Trained denoising model
        signals: Normalized signals of shape (n_signals, signal_length)
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Denoised signals of shape (n_signals, signal_length)
    """
    model.eval()
    n_signals = signals.shape[0]
    # Keep denoising output in float64 to avoid underflow when later
    # restoring ultra-small physical amplitudes.
    denoised = np.zeros(signals.shape, dtype=np.float64)

    print(f"\n[INFO] Denoising {n_signals} signals...")

    with torch.no_grad():
        for i in range(0, n_signals, batch_size):
            end_idx = min(i + batch_size, n_signals)
            batch = signals[i:end_idx]

            # Add channel dimension: (B, L) -> (B, 1, L)
            batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(device)

            # Run inference
            output = model(batch_tensor)

            # Remove channel dimension and move to CPU
            denoised[i:end_idx] = output.squeeze(1).cpu().numpy().astype(np.float64)

            # Progress
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == n_signals:
                print(f"  Processed {end_idx}/{n_signals} signals")

    print(f"[INFO] Denoising complete!")
    return denoised


def denormalize_signals(denoised_normalized: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Denormalize denoised signals back to original scale.

    This function reverses the normalization applied by normalize_signal().
    It properly handles small amplitude signals by using the same dynamic
    threshold logic.

    Args:
        denoised_normalized: Denoised signals in [-1, 1] range
        metadata: Metadata containing normalization parameters

    Returns:
        Denoised signals in original scale
    """
    signal_mins = metadata["signal_mins"]
    signal_maxs = metadata["signal_maxs"]

    # Preserve float64 precision during inverse scaling to prevent tiny signals
    # from being rounded to zero in float32.
    denoised = np.zeros(denoised_normalized.shape, dtype=np.float64)

    for i in range(denoised_normalized.shape[0]):
        # Reverse normalization: normalized = 2 * (x - min) / (max - min) - 1
        # x = (normalized + 1) / 2 * (max - min) + min
        min_val = signal_mins[i]
        max_val = signal_maxs[i]
        amplitude_range = max_val - min_val

        # Use dynamic threshold matching normalize_signal
        # This ensures consistency between normalization and denormalization
        min_threshold = np.finfo(np.float64).eps * 0.1  # ~2.22e-16

        if amplitude_range > min_threshold:
            # Normal denormalization
            denoised[i] = (
                (denoised_normalized[i].astype(np.float64) + 1.0) / 2.0
                * amplitude_range
                + min_val
            )
        else:
            # Signal was nearly flat during normalization
            # For very small signals, preserve the denoised structure
            if amplitude_range > 0:
                # Even tiny signals should be denormalized properly
                denoised[i] = (
                    (denoised_normalized[i].astype(np.float64) + 1.0) / 2.0
                    * amplitude_range
                    + min_val
                )
            else:
                # Truly flat signal (constant value)
                # Restore to the constant value (min_val == max_val)
                denoised[i] = np.full_like(
                    denoised_normalized[i], min_val, dtype=np.float64
                )

    return denoised


def reverse_interpolation(
    signals: np.ndarray, metadata: dict, method: str = "cubic"
) -> np.ndarray:
    """
    Reverse spatial interpolation to get back to original grid size.

    Args:
        signals: Signals at target grid size (n_target_points, signal_length)
        metadata: Metadata containing grid sizes
        method: Interpolation method

    Returns:
        Signals at original grid size
    """
    if not metadata.get("interpolated", False):
        return signals

    input_cols, input_rows = metadata["input_grid_size"]
    target_cols, target_rows = metadata["output_grid_size"]

    print(
        f"\n[INFO] Reverse interpolation: {target_cols}x{target_rows} -> {input_cols}x{input_rows}"
    )

    # Reshape to grid
    signal_length = signals.shape[1]
    grid = reshape_to_grid(signals, target_cols, target_rows)

    # Interpolate back to original size
    grid_small = interpolate_spatial(grid, input_cols, input_rows, method=method)

    # Flatten
    return flatten_grid(grid_small)


def save_to_mat(
    denoised_signals: np.ndarray,
    metadata: dict,
    output_path: str,
    include_original_size: bool = True,
) -> str:
    """
    Save denoised signals to .mat file.

    Args:
        denoised_signals: Denoised signals (at target grid size)
        metadata: Metadata containing original data info
        output_path: Output directory or file path
        include_original_size: Whether to include signals at original grid size

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)

    # If output_path is a directory, create filename with timestamp
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"denoised_{timestamp}.mat"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path

    # Prepare data dict for .mat file
    mat_dict = {
        "x": metadata["time_vector"],  # Time vector
        "y": denoised_signals.T
        if denoised_signals.shape[0] > denoised_signals.shape[1]
        else denoised_signals,
        "signal_length": metadata["signal_length"],
        "grid_cols": metadata["output_grid_size"][0],
        "grid_rows": metadata["output_grid_size"][1],
    }

    # Save
    sio.savemat(output_file, mat_dict)
    print(f"\n[INFO] Saved denoised data to: {output_file}")

    return str(output_file)


def run_inference(
    input_path: str,
    output_path: str,
    checkpoint_path: str = str(CHECKPOINTS_DIR / "best_model.pth"),
    model_type: str = "deep",
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    interpolation_method: str = "cubic",
    batch_size: int = 64,
    save_original_size: bool = True,
    target_signal_length: int = TRAIN_SIGNAL_LENGTH,
    validation_save_path: Optional[str] = None,
) -> str:
    """
    Main inference pipeline.

    Args:
        input_path: Path to input .mat file
        output_path: Output directory or file path
        checkpoint_path: Path to model checkpoint
        model_type: Model architecture
        grid_cols: Input grid columns
        grid_rows: Input grid rows
        target_cols: Target grid columns (model input size)
        target_rows: Target grid rows (model input size)
        interpolation_method: Spatial interpolation method
        batch_size: Inference batch size
        save_original_size: Whether to also save at original grid size
        target_signal_length: Target signal length (truncate if longer)

    Returns:
        Path to saved output file
    """
    print("=" * 60)
    print("Ultrasonic Signal Denoising - Inference")
    print("=" * 60)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = load_model(checkpoint_path, model_type, device)

    # Preprocess input data
    normalized_signals, original_signals, metadata = preprocess_mat_data(
        input_path,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        target_cols=target_cols,
        target_rows=target_rows,
        interpolation_method=interpolation_method,
        target_signal_length=target_signal_length,
    )
    _print_signal_stats("Input (normalized)", normalized_signals)
    _print_signal_stats("Input (original scale)", original_signals)

    # Run denoising
    denoised_normalized = denoise_signals(model, normalized_signals, device, batch_size)
    _print_signal_stats("Model output (normalized)", denoised_normalized)

    # Denormalize
    print("\n[INFO] Denormalizing signals...")
    denoised = denormalize_signals(denoised_normalized, metadata)
    _print_signal_stats("Model output (original scale)", denoised)

    if np.count_nonzero(denoised) == 0 and np.count_nonzero(denoised_normalized) > 0:
        print(
            "[WARNING] Denormalized output is all zeros while normalized output is not. "
            "This usually indicates numerical underflow from tiny amplitudes."
        )

    # Save results
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    comparison_fig = str(
        image_dir
        / (
            "fig_unsupervised_inferenced.png"
            if model_type == "unsupervised_deep"
            else "fig_inferenced.png"
        )
    )
    plot_inference_comparison(original_signals, denoised, comparison_fig)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Acoustic feature validation (声学特征对比: 去噪前 vs 去噪后)
    print("\n" + "=" * 60)
    print("[INFO] Running acoustic feature validation...")
    print("=" * 60)
    if validation_save_path:
        validation_fig = validation_save_path
    else:
        validation_fig = str(image_dir / f"acoustic_validation_{timestamp}.png")
    run_inference_validation(
        input_signals=normalized_signals,
        denoised_signals=denoised_normalized,
        save_path=validation_fig,
    )

    # Save at target grid size (41x41)
    output_file_full = output_dir / f"denoised_{timestamp}_full.mat"
    mat_dict_full = {
        "x": metadata["time_vector"],
        "y": denoised,
        "signal_length": metadata["signal_length"],
        "grid_cols": metadata["output_grid_size"][0],
        "grid_rows": metadata["output_grid_size"][1],
        "n_signals": denoised.shape[0],
    }
    sio.savemat(output_file_full, mat_dict_full)
    print(
        f"\n[INFO] Saved full resolution ({target_cols}x{target_rows}) to: {output_file_full}"
    )

    # Optionally save at original grid size
    if save_original_size and metadata.get("interpolated", False):
        print("\n[INFO] Reverse interpolating to original size...")
        denoised_original_size = reverse_interpolation(
            denoised, metadata, interpolation_method
        )

        output_file_original = output_dir / f"denoised_{timestamp}_original.mat"
        mat_dict_original = {
            "x": metadata["time_vector"],
            "y": denoised_original_size,
            "signal_length": metadata["signal_length"],
            "grid_cols": metadata["input_grid_size"][0],
            "grid_rows": metadata["input_grid_size"][1],
            "n_signals": denoised_original_size.shape[0],
        }
        sio.savemat(output_file_original, mat_dict_original)
        print(
            f"[INFO] Saved original resolution ({grid_cols}x{grid_rows}) to: {output_file_original}"
        )

    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"  Input file: {input_path}")
    print(f"  Output directory: {output_path}")
    print(f"  Model: {model_type}")
    print(f"  Device: {device}")
    print(f"  Acoustic validation: {validation_fig}")

    return str(output_file_full)


def main():
    parser = argparse.ArgumentParser(
        description="Denoise ultrasonic signals using trained CAE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input .mat file (noisy signals)",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for denoised signals",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="deep",
        choices=["lightweight", "deeper", "deep", "unsupervised_deep"],
        help="Model architecture",
    )

    # Grid size arguments
    parser.add_argument(
        "--cols", type=int, default=21, help="Number of columns in input grid"
    )
    parser.add_argument(
        "--rows", type=int, default=21, help="Number of rows in input grid"
    )
    parser.add_argument(
        "--target_cols", type=int, default=41, help="Target columns after interpolation"
    )
    parser.add_argument(
        "--target_rows", type=int, default=41, help="Target rows after interpolation"
    )

    # Processing options
    parser.add_argument(
        "--interp_method",
        type=str,
        default="cubic",
        choices=["linear", "cubic"],
        help="Spatial interpolation method",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Inference batch size"
    )
    parser.add_argument(
        "--no_original_size",
        action="store_true",
        help="Do not save results at original grid size",
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        default=TRAIN_SIGNAL_LENGTH,
        help="Target signal length (truncate if longer)",
    )

    args = parser.parse_args()

    if args.signal_length != TRAIN_SIGNAL_LENGTH:
        raise ValueError(
            f"--signal_length must be {TRAIN_SIGNAL_LENGTH} to match training tensor size"
        )

    run_inference(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        grid_cols=args.cols,
        grid_rows=args.rows,
        target_cols=args.target_cols,
        target_rows=args.target_rows,
        interpolation_method=args.interp_method,
        batch_size=args.batch_size,
        save_original_size=not args.no_original_size,
        target_signal_length=args.signal_length,
    )


if __name__ == "__main__":
    main()
