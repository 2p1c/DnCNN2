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
    uv run python inference.py --input noisy.mat --output results/denoised/
    
    # With custom model checkpoint
    uv run python inference.py --input noisy.mat --output results/ --checkpoint checkpoints/best_model.pth
    
    # Specify grid size (if different from default 21x21)
    uv run python inference.py --input noisy.mat --output results/ --cols 21 --rows 21
"""

import argparse
import numpy as np
import torch
from scipy import io as sio
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

from model import LightweightCAE, DeeperCAE, DeepCAE
from transformer import (
    load_mat_file, 
    reshape_to_grid, 
    flatten_grid,
    interpolate_spatial,
    normalize_signal
)


def load_model(
    checkpoint_path: str,
    model_type: str = 'deep',
    device: torch.device = None
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        model_type: Model architecture ('lightweight', 'deeper', 'deep')
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"[INFO] Loading model from {checkpoint_path}")
    print(f"[INFO] Model type: {model_type}")
    print(f"[INFO] Device: {device}")
    
    # Initialize model based on type
    if model_type == 'deep':
        model = DeepCAE(dropout_rate=0.0)  # No dropout for inference
    elif model_type == 'deeper':
        model = DeeperCAE(dropout_rate=0.0)
    else:
        model = LightweightCAE(dropout_rate=0.0)
    
    # Load checkpoint
    # Note: weights_only=False is needed for checkpoints saved with additional metadata
    # This is safe as long as the checkpoint comes from a trusted source
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully!")
    return model


def preprocess_mat_data(
    mat_path: str,
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    interpolation_method: str = 'cubic'
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
        
    Returns:
        Tuple of:
            - processed_signals: shape (n_signals, signal_length), normalized
            - original_signals: shape (n_signals, signal_length), unnormalized
            - metadata: dict with original data info for reconstruction
    """
    print(f"\n[Step 1] Loading .mat file: {mat_path}")
    time_vector, signal_data = load_mat_file(mat_path)
    
    # Store original data info for reconstruction
    metadata = {
        'time_vector': time_vector,
        'signal_length': len(time_vector),
        'original_shape': signal_data.shape,
        'input_grid_size': (grid_cols, grid_rows),
        'output_grid_size': (target_cols, target_rows),
        'original_min': signal_data.min(),
        'original_max': signal_data.max(),
    }
    
    # Check if interpolation is needed
    n_input_points = grid_cols * grid_rows
    n_target_points = target_cols * target_rows
    
    if signal_data.shape[0] == n_target_points:
        # Data is already at target size, no interpolation needed
        print(f"[INFO] Data already at target size ({target_cols}x{target_rows}), skipping interpolation")
        processed_signals = signal_data.copy()
        metadata['interpolated'] = False
    elif signal_data.shape[0] == n_input_points:
        # Need to interpolate
        print(f"\n[Step 2] Reshaping to {grid_cols}x{grid_rows} grid...")
        input_grid = reshape_to_grid(signal_data, grid_cols, grid_rows)
        
        print(f"\n[Step 3] Interpolating to {target_cols}x{target_rows}...")
        output_grid = interpolate_spatial(
            input_grid, 
            target_cols, 
            target_rows,
            method=interpolation_method
        )
        
        print(f"\n[Step 4] Flattening grid...")
        processed_signals = flatten_grid(output_grid)
        metadata['interpolated'] = True
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
    
    metadata['signal_mins'] = np.array(signal_mins)
    metadata['signal_maxs'] = np.array(signal_maxs)
    
    print(f"[INFO] Preprocessing complete. Output shape: {normalized_signals.shape}")
    
    return normalized_signals, original_signals, metadata


def denoise_signals(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 64
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
    denoised = np.zeros_like(signals)
    
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
            denoised[i:end_idx] = output.squeeze(1).cpu().numpy()
            
            # Progress
            if (i + batch_size) % (batch_size * 10) == 0 or end_idx == n_signals:
                print(f"  Processed {end_idx}/{n_signals} signals")
    
    print(f"[INFO] Denoising complete!")
    return denoised


def denormalize_signals(
    denoised_normalized: np.ndarray,
    metadata: dict
) -> np.ndarray:
    """
    Denormalize denoised signals back to original scale.
    
    Args:
        denoised_normalized: Denoised signals in [-1, 1] range
        metadata: Metadata containing normalization parameters
        
    Returns:
        Denoised signals in original scale
    """
    signal_mins = metadata['signal_mins']
    signal_maxs = metadata['signal_maxs']
    
    denoised = np.zeros_like(denoised_normalized)
    
    for i in range(denoised_normalized.shape[0]):
        # Reverse normalization: normalized = 2 * (x - min) / (max - min) - 1
        # x = (normalized + 1) / 2 * (max - min) + min
        min_val = signal_mins[i]
        max_val = signal_maxs[i]
        
        if max_val - min_val > 1e-10:
            denoised[i] = (denoised_normalized[i] + 1) / 2 * (max_val - min_val) + min_val
        else:
            denoised[i] = np.zeros_like(denoised_normalized[i])
    
    return denoised


def reverse_interpolation(
    signals: np.ndarray,
    metadata: dict,
    method: str = 'cubic'
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
    if not metadata.get('interpolated', False):
        return signals
    
    input_cols, input_rows = metadata['input_grid_size']
    target_cols, target_rows = metadata['output_grid_size']
    
    print(f"\n[INFO] Reverse interpolation: {target_cols}x{target_rows} -> {input_cols}x{input_rows}")
    
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
    include_original_size: bool = True
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
        'x': metadata['time_vector'],  # Time vector
        'y': denoised_signals.T if denoised_signals.shape[0] > denoised_signals.shape[1] else denoised_signals,
        'signal_length': metadata['signal_length'],
        'grid_cols': metadata['output_grid_size'][0],
        'grid_rows': metadata['output_grid_size'][1],
    }
    
    # Save
    sio.savemat(output_file, mat_dict)
    print(f"\n[INFO] Saved denoised data to: {output_file}")
    
    return str(output_file)


def run_inference(
    input_path: str,
    output_path: str,
    checkpoint_path: str = 'checkpoints/best_model.pth',
    model_type: str = 'deep',
    grid_cols: int = 21,
    grid_rows: int = 21,
    target_cols: int = 41,
    target_rows: int = 41,
    interpolation_method: str = 'cubic',
    batch_size: int = 64,
    save_original_size: bool = True
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
        
    Returns:
        Path to saved output file
    """
    print("=" * 60)
    print("Ultrasonic Signal Denoising - Inference")
    print("=" * 60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load model
    model = load_model(checkpoint_path, model_type, device)
    
    # Preprocess input data
    normalized_signals, original_signals, metadata = preprocess_mat_data(
        input_path,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        target_cols=target_cols,
        target_rows=target_rows,
        interpolation_method=interpolation_method
    )
    
    # Run denoising
    denoised_normalized = denoise_signals(model, normalized_signals, device, batch_size)
    
    # Denormalize
    print("\n[INFO] Denormalizing signals...")
    denoised = denormalize_signals(denoised_normalized, metadata)
    
    # Save results
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save at target grid size (41x41)
    output_file_full = output_dir / f"denoised_{timestamp}_full.mat"
    mat_dict_full = {
        'x': metadata['time_vector'],
        'y': denoised,
        'signal_length': metadata['signal_length'],
        'grid_cols': metadata['output_grid_size'][0],
        'grid_rows': metadata['output_grid_size'][1],
        'n_signals': denoised.shape[0],
    }
    sio.savemat(output_file_full, mat_dict_full)
    print(f"\n[INFO] Saved full resolution ({target_cols}x{target_rows}) to: {output_file_full}")
    
    # Optionally save at original grid size
    if save_original_size and metadata.get('interpolated', False):
        print("\n[INFO] Reverse interpolating to original size...")
        denoised_original_size = reverse_interpolation(denoised, metadata, interpolation_method)
        
        output_file_original = output_dir / f"denoised_{timestamp}_original.mat"
        mat_dict_original = {
            'x': metadata['time_vector'],
            'y': denoised_original_size,
            'signal_length': metadata['signal_length'],
            'grid_cols': metadata['input_grid_size'][0],
            'grid_rows': metadata['input_grid_size'][1],
            'n_signals': denoised_original_size.shape[0],
        }
        sio.savemat(output_file_original, mat_dict_original)
        print(f"[INFO] Saved original resolution ({grid_cols}x{grid_rows}) to: {output_file_original}")
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"  Input file: {input_path}")
    print(f"  Output directory: {output_path}")
    print(f"  Model: {model_type}")
    print(f"  Device: {device}")
    
    return str(output_file_full)


def main():
    parser = argparse.ArgumentParser(
        description='Denoise ultrasonic signals using trained CAE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input .mat file (noisy signals)')
    
    # Optional arguments
    parser.add_argument('--output', '-o', type=str, default='results/denoised',
                        help='Output directory for denoised signals')
    parser.add_argument('--checkpoint', '-c', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model', '-m', type=str, default='deep',
                        choices=['lightweight', 'deeper', 'deep'],
                        help='Model architecture')
    
    # Grid size arguments
    parser.add_argument('--cols', type=int, default=21,
                        help='Number of columns in input grid')
    parser.add_argument('--rows', type=int, default=21,
                        help='Number of rows in input grid')
    parser.add_argument('--target_cols', type=int, default=41,
                        help='Target columns after interpolation')
    parser.add_argument('--target_rows', type=int, default=41,
                        help='Target rows after interpolation')
    
    # Processing options
    parser.add_argument('--interp_method', type=str, default='cubic',
                        choices=['linear', 'cubic'],
                        help='Spatial interpolation method')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Inference batch size')
    parser.add_argument('--no_original_size', action='store_true',
                        help='Do not save results at original grid size')
    
    args = parser.parse_args()
    
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
        save_original_size=not args.no_original_size
    )


if __name__ == "__main__":
    main()
