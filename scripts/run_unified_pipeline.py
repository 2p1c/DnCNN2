"""
Unified End-to-End Pipeline for Ultrasonic Denoising.

This script combines the full workflow into one command:
1) Transform raw .mat files into training dataset format
2) Train PINN or DeepSets+PINN model
3) Run inference with the trained checkpoint

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

from scripts.transformer import (
    transform_data,
)
from scripts.analysis.inference import run_inference as run_pinn_inference
from scripts.analysis.inference_deepsets import run_inference as run_deepsets_inference
from scripts.train.train import train_from_config


def _write_experiment_record(
    experiment_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    physics_weight: float,
    checkpoint_path: str | None,
    output_mat: str | None,
    validation_figure: str | None,
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
        f"- Checkpoint: `{checkpoint_path if checkpoint_path else 'N/A (inference skipped)'}`\n"
        f"- Denoised mat: `{output_mat if output_mat else 'N/A (inference skipped)'}`\n"
        f"- Acoustic validation figure: `{validation_figure if validation_figure else 'N/A (inference skipped)'}`\n\n"
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


def _create_run_output_dir(results_dir: str) -> tuple[Path, str]:
    """Create a timestamped run directory under results root."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, timestamp


def _validate_existing_file(path_str: str | None, arg_name: str) -> Path:
    """Validate an input file path and return a resolved Path."""
    if not path_str:
        raise ValueError(f"--{arg_name} is required")

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"--{arg_name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"--{arg_name} must point to a file, got: {path}")
    return path


def _validate_dataset_dir(data_dir: str, require_tf: bool = False) -> None:
    """Validate transformed dataset directory structure for file-mode training."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"--data_dir not found: {root}")
    if not root.is_dir():
        raise ValueError(f"--data_dir must be a directory, got: {root}")

    required_dirs = [
        root / "train" / "noisy",
        root / "train" / "clean",
        root / "val" / "noisy",
        root / "val" / "clean",
    ]
    if require_tf:
        required_dirs.extend([root / "train" / "tf", root / "val" / "tf"])
    missing = [str(p) for p in required_dirs if not p.exists() or not p.is_dir()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Invalid --data_dir structure. Missing directories:\n"
            f"  - {joined}\n"
            "Expected transformed data layout: train/clean, train/noisy, val/clean, val/noisy"
        )


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
        description="Unified pipeline: transform -> train -> inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre_parser],
    )

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
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Resume training from an existing checkpoint",
    )

    parser.add_argument("--skip_transform", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")

    parser.add_argument("--input_cols", type=int, default=21)
    parser.add_argument("--input_rows", type=int, default=21)
    parser.add_argument("--target_cols", type=int, default=41)
    parser.add_argument("--target_rows", type=int, default=41)
    parser.add_argument(
        "--inference_input_cols",
        type=int,
        default=None,
        help="Input grid columns for inference/validation; defaults to input_cols",
    )
    parser.add_argument(
        "--inference_input_rows",
        type=int,
        default=None,
        help="Input grid rows for inference/validation; defaults to input_rows",
    )
    parser.add_argument(
        "--inference_target_cols",
        type=int,
        default=None,
        help="Target grid columns for inference/validation; defaults to target_cols",
    )
    parser.add_argument(
        "--inference_target_rows",
        type=int,
        default=None,
        help="Target grid rows for inference/validation; defaults to target_rows",
    )
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
        "--model_type",
        choices=["deepsets", "tf_fusion"],
        default="deepsets",
        help="DeepSets pipeline model variant",
    )
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--coord_dim", type=int, default=64)
    parser.add_argument("--signal_embed_dim", type=int, default=128)
    parser.add_argument("--coord_embed_dim", type=int, default=64)
    parser.add_argument("--point_dim", type=int, default=128)
    parser.add_argument("--tf_embed_dim", type=int, default=128)
    parser.add_argument("--stft_n_fft", type=int, default=128)
    parser.add_argument("--stft_hop_length", type=int, default=32)
    parser.add_argument("--stft_win_length", type=int, default=128)
    parser.add_argument("--stft_window", type=str, default="hann")
    parser.add_argument(
        "--stft_pooling", choices=["mean", "max", "meanmax"], default="mean"
    )
    parser.add_argument("--fusion_mode", choices=["gated", "concat"], default="gated")
    parser.add_argument("--debug_numerics", action="store_true")

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

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(remaining_argv)

    if not args.skip_inference:
        if not args.inference_input:
            raise ValueError(
                "--inference_input is required unless --skip_inference is enabled"
            )
        _validate_existing_file(args.inference_input, "inference_input")

    if args.pipeline == "pinn" and args.model_type != "deepsets":
        raise ValueError(
            "--model_type=tf_fusion is only valid with --pipeline deepsets"
        )

    if not args.skip_transform:
        _validate_existing_file(args.noisy_mat, "noisy_mat")
        _validate_existing_file(args.clean_mat, "clean_mat")
    else:
        _validate_dataset_dir(
            args.data_dir,
            require_tf=(args.pipeline == "deepsets" and args.model_type == "tf_fusion"),
        )

    results_root = Path(args.results_dir)
    run_dir, run_timestamp = _create_run_output_dir(args.results_dir)
    checkpoints_dir = run_dir / "checkpoints"
    images_dir = run_dir / "images"
    experiment_dir = run_dir / "experiments"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    physics_weight = args.physics_weight
    if physics_weight is None:
        physics_weight = 1e-3 if args.pipeline == "pinn" else 1e-4

    inference_input_cols = (
        args.inference_input_cols
        if args.inference_input_cols is not None
        else args.input_cols
    )
    inference_input_rows = (
        args.inference_input_rows
        if args.inference_input_rows is not None
        else args.input_rows
    )
    inference_target_cols = (
        args.inference_target_cols
        if args.inference_target_cols is not None
        else args.target_cols
    )
    inference_target_rows = (
        args.inference_target_rows
        if args.inference_target_rows is not None
        else args.target_rows
    )

    print("=" * 70)
    print("Unified Ultrasonic Training + Inference Pipeline")
    print("=" * 70)
    print(f"[CONFIG] pipeline={args.pipeline}")
    print(f"[CONFIG] data_dir={args.data_dir}")
    print(f"[CONFIG] results_dir={args.results_dir}")
    print(f"[CONFIG] run_dir={run_dir}")
    print(f"[CONFIG] epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"[CONFIG] physics_weight={physics_weight}")
    print(
        f"[CONFIG] train_grid={args.input_cols}x{args.input_rows} -> {args.target_cols}x{args.target_rows}"
    )
    print(
        "[CONFIG] "
        f"inference_grid={inference_input_cols}x{inference_input_rows} "
        f"-> {inference_target_cols}x{inference_target_rows}"
    )
    print(f"[CONFIG] skip_inference={args.skip_inference}")
    print(f"[CONFIG] signal_length={args.signal_length}, interp={args.interp_method}")

    if not args.skip_transform:
        print("\n[STAGE 1/3] Data transform")
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
            export_tf=(args.pipeline == "deepsets" and args.model_type == "tf_fusion"),
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
            stft_window=args.stft_window,
            stft_pooling=args.stft_pooling,
        )
    else:
        print("\n[STAGE 1/3] Data transform skipped")

    if not args.skip_train:
        print("\n[STAGE 2/3] Model training")
        train_from_config(
            {
                "pipeline": args.pipeline,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "save_best": True,
                "checkpoint_dir": str(checkpoints_dir),
                "seed": args.seed,
                "mode": "file",
                "data_mode": "file",
                "data_path": args.data_dir,
                "patience": args.patience,
                "min_epochs": args.min_epochs,
                "dropout": args.dropout,
                "augment": False if args.pipeline == "pinn" else True,
                "physics_weight": physics_weight,
                "wave_speed": args.wave_speed,
                "center_frequency": args.center_frequency,
                "damping_ratio": args.damping_ratio,
                "grid_cols": args.target_cols,
                "grid_rows": args.target_rows,
                "patch_size": args.patch_size,
                "stride": args.stride,
                "base_channels": args.base_channels,
                "coord_dim": args.coord_dim,
                "signal_embed_dim": args.signal_embed_dim,
                "coord_embed_dim": args.coord_embed_dim,
                "point_dim": args.point_dim,
                "tf_embed_dim": args.tf_embed_dim,
                "model_type": args.model_type,
                "stft_n_fft": args.stft_n_fft,
                "stft_hop_length": args.stft_hop_length,
                "stft_win_length": args.stft_win_length,
                "stft_window": args.stft_window,
                "stft_pooling": args.stft_pooling,
                "fusion_mode": args.fusion_mode,
                "debug_numerics": args.debug_numerics,
                "resume_checkpoint": args.resume_checkpoint,
            }
        )
    else:
        print("\n[STAGE 2/3] Model training skipped")

    checkpoint_path: str | None = None
    output_mat: str | None = None
    val_fig: str | None = None

    if not args.skip_inference:
        checkpoint_name = (
            "best_pinn_model.pth"
            if args.pipeline == "pinn"
            else "best_deepsets_pinn.pth"
        )
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.skip_train:
            checkpoint_path = str(results_root / "checkpoints" / checkpoint_name)
        else:
            checkpoint_path = str(checkpoints_dir / checkpoint_name)

        _validate_existing_file(checkpoint_path, "checkpoint")

        print("\n[STAGE 3/3] Inference")
        if args.pipeline == "pinn":
            val_fig = str(images_dir / f"acoustic_validation_{run_timestamp}.png")
            output_mat = run_pinn_inference(
                input_path=args.inference_input,
                output_path=str(run_dir),
                checkpoint_path=checkpoint_path,
                model_type="deep",
                grid_cols=inference_input_cols,
                grid_rows=inference_input_rows,
                target_cols=inference_target_cols,
                target_rows=inference_target_rows,
                interpolation_method=args.interp_method,
                batch_size=args.inference_batch_size,
                save_original_size=True,
                target_signal_length=args.signal_length,
                validation_save_path=val_fig,
            )
        else:
            val_fig = str(images_dir / f"acoustic_validation_deepsets_{run_timestamp}.png")
            output_mat = run_deepsets_inference(
                input_path=args.inference_input,
                output_path=str(run_dir),
                checkpoint_path=checkpoint_path,
                grid_cols=inference_input_cols,
                grid_rows=inference_input_rows,
                target_cols=inference_target_cols,
                target_rows=inference_target_rows,
                patch_size=args.patch_size,
                model_type=args.model_type,
                fusion_mode=args.fusion_mode,
                debug_numerics=args.debug_numerics,
                interpolation_method=args.interp_method,
                target_signal_length=args.signal_length,
                stft_n_fft=args.stft_n_fft,
                stft_hop_length=args.stft_hop_length,
                stft_win_length=args.stft_win_length,
                stft_window=args.stft_window,
                stft_pooling=args.stft_pooling,
                validation_save_path=val_fig,
            )
    else:
        print("\n[STAGE 3/3] Inference skipped")

    ts = run_timestamp

    print("\n" + "=" * 70)
    print("Pipeline finished successfully")
    print("=" * 70)
    print(f"[RESULT] checkpoint: {checkpoint_path if checkpoint_path else 'N/A'}")
    print(f"[RESULT] denoised mat: {output_mat if output_mat else 'N/A'}")
    print(f"[RESULT] acoustic validation figure: {val_fig if val_fig else 'N/A'}")
    print(f"[RESULT] run directory: {run_dir}")

    if args.log_experiment:
        record_path = _write_experiment_record(
            experiment_dir=experiment_dir,
            timestamp=ts,
            args=args,
            physics_weight=physics_weight,
            checkpoint_path=checkpoint_path,
            output_mat=output_mat,
            validation_figure=val_fig,
        )
        print(f"[RESULT] experiment record: {record_path}")


if __name__ == "__main__":
    main()
