# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ultrasonic-cae** - Ultrasonic 1D signal denoising for NDT workflows using PyTorch. Two active pipelines:
- `pinn`: DeepCAE with Physics-Informed Neural Network constraints
- `deepsets`: DeepSets-PINN architecture

## Setup

```bash
uv sync
uv sync --dev  # with dev dependencies
```

## Common Commands

```bash
# Data preparation
uv run python scripts/transformer.py --noisy data/noisy.mat --clean data/clean.mat --output data

# Training
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --epochs 100
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 100

# Unified pipeline (transform + train + inference)
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_deepsets_template.json

# Inference only
uv run python scripts/analysis/inference.py --input data/noisy.mat --output results/ --checkpoint <path>
uv run python scripts/analysis/inference_deepsets.py --input data/noisy.mat --output results/ --checkpoint <path>

# Validation
uv run python scripts/analysis/acoustic_validation.py
uv run python scripts/analysis/preview_signals.py --detailed --no_show
```

## Architecture

- **Model definition**: `model/model.py` - Contains `DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN`
- **Training entry**: `scripts/train/train.py` - Unified trainer with `--pipeline` flag
- **Unified flow**: `scripts/run_unified_pipeline.py` - End-to-end pipeline orchestrator
- **Data transform**: `scripts/transformer.py` - Converts .mat files to train/val directory structure
- **Inference**: `scripts/analysis/inference.py` (PINN), `scripts/analysis/inference_deepsets.py` (DeepSets)
- **Configs**: `configs/` - JSON templates for pipeline parameters

## Key Physics Constants

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 6.25 MHz |
| Duration | 160 μs |
| Points | 1000 |
| Center Frequency | 250 kHz |
| Wave Speed | 5900 m/s (steel) |

## Output Structure

Unified pipeline creates `results/<timestamp>/` containing:
- `checkpoints/` - Model weights
- `images/` - Training visualizations
- `experiments/` - Markdown experiment records (when `--log_experiment`)

## Workflow Notes

- `--skip_transform` requires existing `data/train/{noisy,clean}` and `data/val/{noisy,clean}` directories
- `--skip_train` with `--checkpoint` enables inference-only runs
- `physics_weight` defaults: PINN=`1e-3`, DeepSets=`1e-4`
- Device priority: cuda > mps > cpu
- Signal length is enforced at 1000 points

## Verification

```bash
uv run python -m compileall .
uv build
```
