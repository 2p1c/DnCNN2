# Ultrasonic Signal Denoising (CAE / PINN / DeepSets-PINN)

Ultrasonic 1D signal denoising project for NDT workflows, with three training pipelines:

- `CAE` baseline
- `PINN` (physics-informed loss)
- `DeepSets + PINN` (patch/set-based spatial modeling)

The project supports synthetic data generation, file-based training, acoustic feature validation, and `.mat` inference export.

## Environment

```bash
# Install dependencies
uv sync
```

## Standard Output Directories

All scripts now follow this unified convention by default:

- Figures: `results/images/`
- Checkpoints: `results/checkpoints/`
- Inference `.mat` outputs: `results/`

Directories are auto-created when needed.

## Quick Start

```bash
# Baseline CAE training
uv run python scripts/train/train.py

# PINN training
uv run python scripts/train/train_pinn.py

# DeepSets + PINN training
uv run python scripts/train/train_deepsets_pinn.py --data_path data
```

## Training Commands

### 1. Baseline CAE

```bash
# Synthetic data (default)
uv run python scripts/train/train.py

# File mode
uv run python scripts/train/train.py --mode file --data_path data

# Deep model + longer training
uv run python scripts/train/train.py --model deep --epochs 200 --lr 5e-4
```

Default checkpoint:
- `results/checkpoints/best_model.pth`

Typical figures:
- `results/images/fig_pre_train_samples.png`
- `results/images/fig_results.png`
- `results/images/fig_training_curves.png`
- `results/images/fig_acoustic_validation.png`

### 2. PINN (CAE + physics)

```bash
uv run python scripts/train/train_pinn.py
uv run python scripts/train/train_pinn.py --mode file --data_path data --physics_weight 0.001
```

Default checkpoint:
- `results/checkpoints/best_pinn_model.pth`

Typical figures:
- `results/images/fig_pinn_pre_train_samples.png`
- `results/images/fig_pinn_results.png`
- `results/images/fig_pinn_training_curves.png`
- `results/images/fig_pinn_acoustic_validation.png`

### 3. DeepSets + PINN

```bash
uv run python scripts/train/train_deepsets_pinn.py --data_path data
uv run python scripts/train/train_deepsets_pinn.py --epochs 100 --patch_size 7 --physics_weight 1e-4
```

模型命名新旧对照（兼容）：

| 维度 | 旧名称 | 新名称（推荐） | 兼容状态 |
|---|---|---|---|
| Python 类名 | `DeepSetsPINN` | `SetInvariantWavePINN` | 两者都可导入/使用 |
| Python 类名 | `SpatialAuxiliaryCAE` | `SpatialContextCAE` | 两者都可导入/使用 |
| CLI `--model_type` | `deepsets` | `set_invariant_pinn` | 两者都可用 |
| CLI `--model_type` | `spatial_cae` | `spatial_context_cae` | 两者都可用 |

推荐写法示例：

```bash
# 推荐：新命名（语义更清晰）
uv run python scripts/train/train_deepsets_pinn.py --model_type spatial_context_cae --data_path data
uv run python scripts/train/train_deepsets_pinn.py --model_type set_invariant_pinn --data_path data

# 兼容：旧命名（仍可用）
uv run python scripts/train/train_deepsets_pinn.py --model_type spatial_cae --data_path data
uv run python scripts/train/train_deepsets_pinn.py --model_type deepsets --data_path data
```

Default checkpoint:
- `results/checkpoints/best_deepsets_pinn.pth`

Typical figures:
- `results/images/fig_deepsets_pinn_training_curves.png`
- `results/images/fig_deepsets_pinn_results.png`

## Inference

### CAE / PINN Inference

```bash
uv run python scripts/analysis/inference.py \
	--input noisy.mat \
	--output results/ \
	--checkpoint results/checkpoints/best_model.pth
```

Outputs:
- `results/denoised_<timestamp>_full.mat`
- `results/denoised_<timestamp>_original.mat` (if reverse interpolation enabled)
- `results/images/acoustic_validation_<timestamp>.png`

### DeepSets Inference

```bash
uv run python scripts/analysis/inference_deepsets.py \
	--input noisy.mat \
	--output results/ \
	--checkpoint results/checkpoints/best_deepsets_pinn.pth
```

Outputs:
- `results/deepsets_denoised_<timestamp>.mat`
- `results/deepsets_denoised_<timestamp>_original.mat` (if applicable)

## Signal Preview Utilities

```bash
# Basic preview
uv run python scripts/analysis/preview_signals.py

# With detailed plot + noise type comparison
uv run python scripts/analysis/preview_signals.py --detailed --compare_noise --no_show
```

Preview figures are saved under `results/images/` by default.

## Core Signal Settings

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 6.25 MHz |
| Duration | 160 us |
| Data Points | 1000 |
| Center Frequency | 250 kHz |

## Project Structure (Simplified)

```text
DnCNN2/
├── model/
|   ├── model.py
|   └── model_deepsets.py
├── data/
|   ├── data_utils.py
|   └── data_deepsets.py
├── scripts/
|   ├── transformer.py
|   ├── train/
|   |   ├── train.py
|   |   ├── train_pinn.py
|   |   └── train_deepsets_pinn.py
|   └── analysis/
|       ├── inference.py
|       ├── inference_deepsets.py
|       ├── acoustic_validation.py
|       └── preview_signals.py
├── results/
|   ├── checkpoints/
|   └── images/
└── README.md
```

## License

MIT
