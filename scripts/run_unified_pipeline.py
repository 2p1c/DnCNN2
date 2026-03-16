"""
Unified End-to-End Pipeline for Ultrasonic Denoising.

This script combines the full workflow into one command:
1) Transform raw .mat files into training dataset format
2) Train PINN or DeepSets+PINN model
3) Run inference with the trained checkpoint
4) Run acoustic validation (before vs after denoising)

==========================
Configuration Quick Guide
==========================

Required files you need to prepare:
- noisy_mat: noisy experimental .mat (typically 21x21 points)
- clean_mat: clean reference .mat (typically 41x41 points), needed for transform
- inference_input: .mat used for inference after training

Core selection:
- --pipeline pinn
  Use DeepCAE_PINN training + CAE inference script.
- --pipeline deepsets
  Use DeepSets PINN training + DeepSets inference script.

Recommended defaults:
- grid: input 21x21, target 41x41
- signal_length: 1000
- interp_method: cubic
- physics_weight:
    - pinn:     1e-3
    - deepsets: 1e-4

Typical usage:
- Full PINN pipeline:
  uv run python scripts/run_unified_pipeline.py \
      --pipeline pinn \
      --noisy_mat data/noisy.mat \
      --clean_mat data/clean.mat \
      --inference_input data/noisy.mat

- Full DeepSets pipeline:
  uv run python scripts/run_unified_pipeline.py \
      --pipeline deepsets \
      --noisy_mat data/noisy.mat \
      --clean_mat data/clean.mat \
      --inference_input data/noisy.mat \
      --patch_size 5

- Reuse existing dataset/checkpoint (skip transform and train):
  uv run python scripts/run_unified_pipeline.py \
      --pipeline pinn \
      --inference_input data/noisy.mat \
      --skip_transform \
      --skip_train \
      --checkpoint results/checkpoints/best_pinn_model.pth
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from scripts.transformer import (
    flatten_grid,
    interpolate_spatial,
    load_mat_file,
    normalize_signal,
    reshape_to_grid,
    transform_data,
    truncate_signals,
)
from scripts.analysis.acoustic_validation import run_inference_validation
from scripts.analysis.inference import run_inference as run_pinn_inference
from scripts.analysis.inference_deepsets import run_inference as run_deepsets_inference
from scripts.train.train_deepsets_pinn import train_deepsets_pinn
from scripts.train.train_pinn import train_pinn


def _load_and_align_signals(
    input_mat: str,
    denoised_mat: str,
    input_cols: int,
    input_rows: int,
    target_cols: int,
    target_rows: int,
    signal_length: int,
    interp_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load input/denoised .mat and align them to same shape for validation."""
    _, input_signals = load_mat_file(input_mat)
    _, denoised_signals = load_mat_file(denoised_mat)

    _, input_signals = truncate_signals(
        np.arange(input_signals.shape[1]), input_signals, signal_length
    )
    _, denoised_signals = truncate_signals(
        np.arange(denoised_signals.shape[1]), denoised_signals, signal_length
    )

    in_points = input_cols * input_rows
    tgt_points = target_cols * target_rows

    if input_signals.shape[0] != denoised_signals.shape[0]:
        if (
            input_signals.shape[0] == in_points
            and denoised_signals.shape[0] == tgt_points
        ):
            input_grid = reshape_to_grid(input_signals, input_cols, input_rows)
            input_interp = interpolate_spatial(
                input_grid, target_cols, target_rows, method=interp_method
            )
            input_signals = flatten_grid(input_interp)
        else:
            raise ValueError(
                "Cannot align validation signals: "
                f"input points={input_signals.shape[0]}, "
                f"denoised points={denoised_signals.shape[0]}"
            )

    input_norm = np.zeros_like(input_signals, dtype=np.float32)
    denoised_norm = np.zeros_like(denoised_signals, dtype=np.float32)

    for i in range(input_signals.shape[0]):
        input_norm[i] = normalize_signal(input_signals[i])
        denoised_norm[i] = normalize_signal(denoised_signals[i])

    return input_norm, denoised_norm


def _write_experiment_record(
    experiment_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    physics_weight: float,
    checkpoint_path: str,
    output_mat: str,
    validation_figure: str,
) -> Path:
    """Write one experiment markdown report."""
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.experiment_tag}" if args.experiment_tag else ""
    record_path = experiment_dir / f"exp_{timestamp}_{args.pipeline}{tag}.md"

    args_dict = vars(args).copy()
    args_dict["physics_weight"] = physics_weight

    content = (
        f"# Experiment Record: {timestamp}\n\n"
        f"## Run Summary\n"
        f"- Pipeline: `{args.pipeline}`\n"
        f"- Timestamp: `{timestamp}`\n"
        f"- Config file: `{args.config}`\n"
        f"- Transform skipped: `{args.skip_transform}`\n"
        f"- Train skipped: `{args.skip_train}`\n\n"
        f"## Inputs\n"
        f"- Noisy mat: `{args.noisy_mat}`\n"
        f"- Clean mat: `{args.clean_mat}`\n"
        f"- Inference input: `{args.inference_input}`\n\n"
        f"## Outputs\n"
        f"- Checkpoint: `{checkpoint_path}`\n"
        f"- Denoised mat: `{output_mat}`\n"
        f"- Acoustic validation figure: `{validation_figure}`\n\n"
        f"## Full Runtime Config\n"
        "```json\n"
        f"{json.dumps(args_dict, indent=2)}\n"
        "```\n\n"
        "## Metrics (Fill After Review)\n"
        "- Best val PSNR: \n"
        "- Final train PSNR: \n"
        "- Physics loss trend: \n"
        "- Acoustic validation verdict: \n\n"
        "## Notes\n"
        "- What changed vs previous run:\n"
        "- Observed failure modes:\n"
        "- Next run plan:\n"
    )

    record_path.write_text(content, encoding="utf-8")
    return record_path


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("Config file must be a JSON object at top level")
        config_defaults = loaded

    parser = argparse.ArgumentParser(
        description="Unified pipeline: transform -> train -> inference -> acoustic validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )

    if config_defaults:
        parser.set_defaults(**config_defaults)

    parser.add_argument("--pipeline", choices=["pinn", "deepsets"], default="pinn")

    parser.add_argument(
        "--noisy_mat", type=str, help="Path to noisy .mat for transform"
    )
    parser.add_argument(
        "--clean_mat", type=str, help="Path to clean .mat for transform"
    )
    parser.add_argument(
        "--inference_input", type=str, default=None, help="Input .mat for inference"
    )

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Override checkpoint path"
    )

    parser.add_argument("--skip_transform", action="store_true")
    parser.add_argument("--skip_train", action="store_true")

    parser.add_argument("--input_cols", type=int, default=21)
    parser.add_argument("--input_rows", type=int, default=21)
    parser.add_argument("--target_cols", type=int, default=41)
    parser.add_argument("--target_rows", type=int, default=41)
    parser.add_argument("--signal_length", type=int, default=1000)
    parser.add_argument("--interp_method", choices=["linear", "cubic"], default="cubic")
    parser.add_argument("--augment_factor", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--physics_weight", type=float, default=None)

    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--model_type", choices=["spatial_cae", "deepsets"], default="spatial_cae"
    )
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--coord_dim", type=int, default=64)

    parser.add_argument("--wave_speed", type=float, default=5900.0)
    parser.add_argument("--center_frequency", type=float, default=250e3)
    parser.add_argument("--damping_ratio", type=float, default=0.05)

    parser.add_argument("--inference_batch_size", type=int, default=64)
    parser.add_argument("--validation_samples", type=int, default=20)

    parser.add_argument(
        "--log_experiment",
        action="store_true",
        help="Write markdown experiment record for this run",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Directory to save experiment records",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="Optional short tag appended to experiment filename",
    )

    args = parser.parse_args(remaining_argv)

    if not args.inference_input:
        raise ValueError(
            "--inference_input is required (can also be provided in --config JSON)"
        )

    results_dir = Path(args.results_dir)
    checkpoints_dir = results_dir / "checkpoints"
    images_dir = results_dir / "images"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    physics_weight = args.physics_weight
    if physics_weight is None:
        physics_weight = 1e-3 if args.pipeline == "pinn" else 1e-4

    print("=" * 70)
    print("Unified Ultrasonic Training + Inference Pipeline")
    print("=" * 70)
    print(f"[CONFIG] pipeline={args.pipeline}")
    print(f"[CONFIG] data_dir={args.data_dir}")
    print(f"[CONFIG] results_dir={args.results_dir}")
    print(f"[CONFIG] epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"[CONFIG] physics_weight={physics_weight}")
    print(
        f"[CONFIG] grid={args.input_cols}x{args.input_rows} -> {args.target_cols}x{args.target_rows}"
    )
    print(f"[CONFIG] signal_length={args.signal_length}, interp={args.interp_method}")

    if not args.skip_transform:
        if not args.noisy_mat or not args.clean_mat:
            raise ValueError(
                "--noisy_mat and --clean_mat are required unless --skip_transform is used"
            )

        print("\n[STAGE 1/4] Data transform")
        transform_data(
            noisy_path=args.noisy_mat,
            clean_path=args.clean_mat,
            output_dir=args.data_dir,
            noisy_grid_size=(args.input_cols, args.input_rows),
            clean_grid_size=(args.target_cols, args.target_rows),
            augment_factor=args.augment_factor,
            train_ratio=args.train_ratio,
            interpolation_method=args.interp_method,
            normalize=True,
            seed=args.seed,
            target_signal_length=args.signal_length,
        )
    else:
        print("\n[STAGE 1/4] Data transform skipped")

    if not args.skip_train:
        print("\n[STAGE 2/4] Model training")
        if args.pipeline == "pinn":
            train_pinn(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_best=True,
                checkpoint_dir=str(checkpoints_dir),
                seed=args.seed,
                data_mode="file",
                data_path=args.data_dir,
                early_stopping_patience=args.patience,
                min_epochs=args.min_epochs,
                dropout_rate=args.dropout,
                augment=False,
                physics_weight=physics_weight,
                wave_speed=args.wave_speed,
                center_frequency=args.center_frequency,
                damping_ratio=args.damping_ratio,
            )
        else:
            train_deepsets_pinn(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_best=True,
                checkpoint_dir=str(checkpoints_dir),
                seed=args.seed,
                data_path=args.data_dir,
                early_stopping_patience=args.patience,
                min_epochs=args.min_epochs,
                dropout_rate=args.dropout,
                augment=True,
                grid_cols=args.target_cols,
                grid_rows=args.target_rows,
                patch_size=args.patch_size,
                stride=args.stride,
                physics_weight=physics_weight,
                wave_speed=args.wave_speed,
                center_frequency=args.center_frequency,
                model_type=args.model_type,
                base_channels=args.base_channels,
                coord_dim=args.coord_dim,
            )
    else:
        print("\n[STAGE 2/4] Model training skipped")

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_name = (
            "best_pinn_model.pth"
            if args.pipeline == "pinn"
            else "best_deepsets_pinn.pth"
        )
        checkpoint_path = str(checkpoints_dir / checkpoint_name)

    print("\n[STAGE 3/4] Inference")
    if args.pipeline == "pinn":
        output_mat = run_pinn_inference(
            input_path=args.inference_input,
            output_path=args.results_dir,
            checkpoint_path=checkpoint_path,
            model_type="deep",
            grid_cols=args.input_cols,
            grid_rows=args.input_rows,
            target_cols=args.target_cols,
            target_rows=args.target_rows,
            interpolation_method=args.interp_method,
            batch_size=args.inference_batch_size,
            save_original_size=True,
            target_signal_length=args.signal_length,
        )
    else:
        output_mat = run_deepsets_inference(
            input_path=args.inference_input,
            output_path=args.results_dir,
            checkpoint_path=checkpoint_path,
            grid_cols=args.input_cols,
            grid_rows=args.input_rows,
            target_cols=args.target_cols,
            target_rows=args.target_rows,
            patch_size=args.patch_size,
            interpolation_method=args.interp_method,
            target_signal_length=args.signal_length,
        )

    print("\n[STAGE 4/4] Acoustic validation")
    input_norm, denoised_norm = _load_and_align_signals(
        input_mat=args.inference_input,
        denoised_mat=output_mat,
        input_cols=args.input_cols,
        input_rows=args.input_rows,
        target_cols=args.target_cols,
        target_rows=args.target_rows,
        signal_length=args.signal_length,
        interp_method=args.interp_method,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_fig = (
        images_dir / f"unified_inference_acoustic_validation_{args.pipeline}_{ts}.png"
    )
    run_inference_validation(
        input_signals=input_norm,
        denoised_signals=denoised_norm,
        save_path=str(val_fig),
        num_samples=args.validation_samples,
    )

    print("\n" + "=" * 70)
    print("Pipeline finished successfully")
    print("=" * 70)
    print(f"[RESULT] checkpoint: {checkpoint_path}")
    print(f"[RESULT] denoised mat: {output_mat}")
    print(f"[RESULT] acoustic validation figure: {val_fig}")

    if args.log_experiment:
        record_path = _write_experiment_record(
            experiment_dir=Path(args.experiment_dir),
            timestamp=ts,
            args=args,
            physics_weight=physics_weight,
            checkpoint_path=checkpoint_path,
            output_mat=output_mat,
            validation_figure=str(val_fig),
        )
        print(f"[RESULT] experiment record: {record_path}")


if __name__ == "__main__":
    main()
