# Architecture

## Design Patterns

### Two Parallel Pipeline Architecture

The project implements two distinct denoising pipelines with shared components:

1. **PINN Pipeline** (`pinn`): DeepCAE with Physics-Informed Neural Network constraints
2. **DeepSets Pipeline** (`deepsets`): DeepSets-PINN architecture

Both pipelines share:
- Data loading and preprocessing (`data/data_utils.py`, `data/data_deepsets.py`)
- Inference infrastructure (`scripts/analysis/inference.py`, `scripts/analysis/inference_deepsets.py`)
- Unified pipeline orchestrator (`scripts/run_unified_pipeline.py`)

### Encoder-Decoder Architecture

All models follow a symmetric encoder-decoder structure:

```
Input (1000 points)
    ↓
Encoder (conv layers, decreasing width)
    ↓
Bottleneck
    ↓
Decoder (deconv layers, increasing width)
    ↓
Output (1000 points)
```

### Physics-Informed Loss

Both pipelines enforce physics constraints via custom loss functions:

- **PINN** (`scripts/train/train.py:148-158`): 1D damped wave equation residual
- **DeepSets** (`scripts/train/train.py:160-175`): 2D wave equation residual

### Data Flow

```
.mat files (noisy/clean)
    ↓
scripts/transformer.py
    ↓
data/{train,val}/{noisy,clean}/ directories
    ↓
data/data_utils.py / data/data_deepsets.py (PyTorch datasets)
    ↓
scripts/train/train.py (training loop)
    ↓
checkpoints/*.pth
    ↓
scripts/analysis/inference.py / inference_deepsets.py
    ↓
.mat output files
```

## Model Architectures

### DeepCAE (base model)

- 6 convolutional blocks (encoder)
- 6 deconvolutional blocks (decoder)
- Batch normalization and LeakyReLU
- Skip connections between corresponding encoder/decoder layers

### DeepCAE_PINN

- Inherits DeepCAE architecture
- Adds coordinate encoding MLP for physics residual computation
- PINN loss computed on equally-spaced grid points

### DeepSetsPINN

- Point-wise MLP encoding (shared weights across signal positions)
- Aggregation via set function (max pooling)
- Grid-wise MLP decoding
- 2D wave equation physics residual (on 2D grid of signal × coordinate)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 1000 fixed signal length | Enforces consistent tensor shapes across all operations |
| Separate train/val split | Default 80/20 split in transformer.py |
| Physics weight defaults (PINN=1e-3, DeepSets=1e-4) | Empirical tuning, DeepSets needs lower weight due to aggregation |
| Device priority: cuda > mps > cpu | GPU acceleration when available |

## Layer Responsibilities

| Layer | Purpose |
|-------|---------|
| `scripts/transformer.py` | .mat → directory structure conversion |
| `data/data_utils.py` | PINN dataset (1D signal loading, augmentation) |
| `data/data_deepsets.py` | DeepSets dataset (2D grid construction) |
| `scripts/train/train.py` | Unified training loop, loss computation, checkpointing |
| `model/model.py` | DeepCAE, DeepCAE_PINN, DeepSetsPINN definitions |
| `scripts/analysis/inference.py` | PINN inference with RRMSE metric |
| `scripts/analysis/inference_deepsets.py` | DeepSets inference with batch processing |
