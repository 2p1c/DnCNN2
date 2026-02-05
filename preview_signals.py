"""
Preview Signal Generation

Quick script to visualize generated signals without running full training.
Use this to tune signal and noise parameters before training.

Usage:
    uv run python preview_signals.py
    uv run python preview_signals.py --num_samples 6 --seed 123
    uv run python preview_signals.py --snr -10 -6 -3 --noise_intensity 2.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import torch

from data_utils import UltrasonicDataset


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def plot_signal_samples(
    dataset: UltrasonicDataset,
    num_samples: int = 6,
    save_path: str = "fig_pre_train_samples.png",
    show_plot: bool = True
) -> None:
    """
    Plot noisy vs clean signal samples.
    
    Args:
        dataset: UltrasonicDataset instance
        num_samples: Number of samples to plot
        save_path: Path to save the figure
        show_plot: Whether to display the plot interactively
    """
    # Time axis (microseconds)
    time_us = np.linspace(0, 160, 1000)
    
    # Calculate grid dimensions
    cols = 2
    rows = num_samples
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.5 * rows))
    fig.suptitle(
        f'Signal Preview: Noisy Input vs Clean Ground Truth\n'
        f'SNR Range: {dataset.snr_range} dB | Noise Types: {dataset.noise_types}\n'
        f'Pulse Bursts: {dataset.num_pulse_bursts_range} | Cycles/Burst: {dataset.cycles_per_burst_range}',
        fontsize=12, fontweight='bold'
    )
    
    for i in range(num_samples):
        noisy, clean = dataset[i]
        noisy_np = noisy[0].numpy()
        clean_np = clean[0].numpy()
        
        # Calculate actual SNR
        noise = noisy_np - clean_np
        actual_snr = calculate_snr(clean_np, noise)
        
        # Plot noisy signal
        ax_noisy = axes[i, 0] if rows > 1 else axes[0]
        ax_noisy.plot(time_us, noisy_np, 'b-', linewidth=0.6, alpha=0.9)
        ax_noisy.set_title(f'Sample {i+1}: Noisy Input (SNR: {actual_snr:.1f} dB)')
        ax_noisy.set_xlabel('Time (μs)')
        ax_noisy.set_ylabel('Amplitude')
        ax_noisy.grid(True, alpha=0.3)
        ax_noisy.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        
        # Auto-scale Y axis based on signal range
        y_max = max(np.abs(noisy_np).max(), np.abs(clean_np).max()) * 1.2
        ax_noisy.set_ylim(-y_max, y_max)
        
        # Plot clean signal
        ax_clean = axes[i, 1] if rows > 1 else axes[1]
        ax_clean.plot(time_us, clean_np, 'g-', linewidth=0.8)
        ax_clean.set_title(f'Sample {i+1}: Clean Ground Truth')
        ax_clean.set_xlabel('Time (μs)')
        ax_clean.set_ylabel('Amplitude')
        ax_clean.grid(True, alpha=0.3)
        ax_clean.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax_clean.set_ylim(-y_max, y_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved preview to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_detailed_comparison(
    dataset: UltrasonicDataset,
    sample_idx: int = 0,
    save_path: str = "fig_detailed_signal.png",
    show_plot: bool = True
) -> None:
    """
    Plot detailed comparison of a single signal with noise breakdown.
    
    Args:
        dataset: UltrasonicDataset instance
        sample_idx: Index of sample to analyze
        save_path: Path to save the figure
        show_plot: Whether to display the plot
    """
    noisy, clean = dataset[sample_idx]
    noisy_np = noisy[0].numpy()
    clean_np = clean[0].numpy()
    noise_np = noisy_np - clean_np
    
    time_us = np.linspace(0, 160, 1000)
    
    # Calculate metrics
    actual_snr = calculate_snr(clean_np, noise_np)
    noise_std = np.std(noise_np)
    signal_max = np.abs(clean_np).max()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(
        f'Detailed Signal Analysis\n'
        f'SNR: {actual_snr:.1f} dB | Noise σ: {noise_std:.4f} | Signal Peak: {signal_max:.4f}',
        fontsize=12, fontweight='bold'
    )
    
    # 1. Clean signal
    axes[0].plot(time_us, clean_np, 'g-', linewidth=0.8)
    axes[0].set_title('Clean Signal (Ground Truth)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.5, 1.5)
    
    # 2. Noise only
    axes[1].plot(time_us, noise_np, 'r-', linewidth=0.5, alpha=0.8)
    axes[1].set_title(f'Noise Component (σ = {noise_std:.4f})')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Noisy signal
    axes[2].plot(time_us, noisy_np, 'b-', linewidth=0.6, alpha=0.9)
    axes[2].set_title(f'Noisy Signal (Clean + Noise)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Overlay comparison
    axes[3].plot(time_us, clean_np, 'g-', linewidth=1.2, label='Clean', alpha=0.8)
    axes[3].plot(time_us, noisy_np, 'b-', linewidth=0.5, label='Noisy', alpha=0.6)
    axes[3].set_title('Overlay Comparison')
    axes[3].set_xlabel('Time (μs)')
    axes[3].set_ylabel('Amplitude')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved detailed analysis to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_noise_type_comparison(
    num_samples_per_type: int = 2,
    snr_db: float = -6.0,
    noise_intensity: float = 1.5,
    save_path: str = "fig_noise_types.png",
    show_plot: bool = True,
    seed: int = 42
) -> None:
    """
    Compare different noise types side by side.
    
    Args:
        num_samples_per_type: Samples per noise type
        snr_db: Fixed SNR for comparison
        noise_intensity: Noise intensity multiplier
        save_path: Path to save figure
        show_plot: Whether to display
        seed: Random seed
    """
    noise_types = ['gaussian', 'pink', 'impulse', 'periodic', 'bandlimited', 'mixed']
    time_us = np.linspace(0, 160, 1000)
    
    fig, axes = plt.subplots(len(noise_types), 2, figsize=(14, 2.5 * len(noise_types)))
    fig.suptitle(
        f'Noise Type Comparison (SNR: {snr_db} dB, Intensity: {noise_intensity}x)',
        fontsize=12, fontweight='bold'
    )
    
    for row, ntype in enumerate(noise_types):
        # Create dataset with single noise type
        dataset = UltrasonicDataset(
            num_samples=1,
            snr_range=(snr_db,),
            noise_types=(ntype,),
            noise_intensity=noise_intensity,
            seed=seed + row
        )
        
        noisy, clean = dataset[0]
        noisy_np = noisy[0].numpy()
        clean_np = clean[0].numpy()
        noise_np = noisy_np - clean_np
        
        actual_snr = calculate_snr(clean_np, noise_np)
        
        # Noisy signal
        axes[row, 0].plot(time_us, noisy_np, 'b-', linewidth=0.6)
        axes[row, 0].set_title(f'{ntype.upper()}: Noisy (SNR: {actual_snr:.1f} dB)')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].grid(True, alpha=0.3)
        
        # Clean signal
        axes[row, 1].plot(time_us, clean_np, 'g-', linewidth=0.8)
        axes[row, 1].set_title(f'{ntype.upper()}: Clean')
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (μs)')
    axes[-1, 1].set_xlabel('Time (μs)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved noise comparison to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Preview ultrasonic signal generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Signal parameters
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # SNR parameters
    parser.add_argument('--snr', type=float, nargs='+', default=[-10.0, -6.0, -3.0, 0.0],
                        help='SNR values in dB (space separated)')
    
    # Noise parameters
    parser.add_argument('--noise_intensity', type=float, default=1.5,
                        help='Noise intensity multiplier (>1 for stronger noise)')
    parser.add_argument('--noise_types', type=str, nargs='+', 
                        default=['gaussian', 'pink', 'impulse', 'periodic', 'bandlimited', 'mixed'],
                        help='Noise types to use')
    
    # Signal structure
    parser.add_argument('--bursts', type=int, nargs=2, default=[1, 4],
                        help='Min and max pulse bursts per signal')
    parser.add_argument('--cycles', type=int, nargs=2, default=[3, 12],
                        help='Min and max carrier cycles per burst')
    
    # Output options
    parser.add_argument('--output', type=str, default='fig_pre_train_samples.png',
                        help='Output filename')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot (just save)')
    parser.add_argument('--detailed', action='store_true',
                        help='Also generate detailed single-sample analysis')
    parser.add_argument('--compare_noise', action='store_true',
                        help='Generate noise type comparison plot')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Signal Generation Preview")
    print("="*60)
    print(f"SNR Range: {args.snr} dB")
    print(f"Noise Intensity: {args.noise_intensity}x")
    print(f"Noise Types: {args.noise_types}")
    print(f"Pulse Bursts: {args.bursts[0]}-{args.bursts[1]}")
    print(f"Cycles/Burst: {args.cycles[0]}-{args.cycles[1]}")
    print(f"Random Seed: {args.seed}")
    print("="*60 + "\n")
    
    # Create dataset
    dataset = UltrasonicDataset(
        num_samples=args.num_samples,
        snr_range=tuple(args.snr),
        num_pulse_bursts_range=tuple(args.bursts),
        cycles_per_burst_range=tuple(args.cycles),
        noise_types=tuple(args.noise_types),
        noise_intensity=args.noise_intensity,
        seed=args.seed
    )
    
    # Generate main preview
    plot_signal_samples(
        dataset,
        num_samples=args.num_samples,
        save_path=args.output,
        show_plot=not args.no_show
    )
    
    # Generate detailed analysis if requested
    if args.detailed:
        plot_detailed_comparison(
            dataset,
            sample_idx=0,
            save_path="fig_detailed_signal.png",
            show_plot=not args.no_show
        )
    
    # Generate noise comparison if requested
    if args.compare_noise:
        plot_noise_type_comparison(
            snr_db=args.snr[0] if args.snr else -6.0,
            noise_intensity=args.noise_intensity,
            save_path="fig_noise_types.png",
            show_plot=not args.no_show,
            seed=args.seed
        )
    
    print("\n[INFO] Preview complete!")


if __name__ == "__main__":
    main()
