# Technology Stack

**Analysis Date:** 2026-04-12

## Languages

**Primary:**
- Python >= 3.10 - All model training, data processing, and inference code

## Runtime

**Environment:**
- Python 3.12 (via `.venv` virtual environment)

**Package Manager:**
- uv >= 0.5.0 (configured in `pyproject.toml`)
- Lockfile: `uv.lock` present

## Frameworks

**Core:**
- PyTorch >= 2.0.0 - Neural network framework for `DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN` models
- torch.nn - Neural network modules (Conv1d, ConvTranspose1d, BatchNorm1d, etc.)

**Scientific Computing:**
- numpy >= 1.24.0, < 2.0.0 - Array operations, signal data handling
- scipy >= 1.10.0 - Signal processing (`scipy.signal`), file I/O (`scipy.io`), interpolation

**Visualization:**
- matplotlib >= 3.7.0 - Training plots, signal visualization

**Progress Tracking:**
- tqdm >= 4.65.0 - Training progress bars

**Configuration:**
- pyyaml - Pipeline configuration parsing (JSON templates)

## Build System

- hatchling - Package build backend defined in `pyproject.toml`
- Package name: `ultrasonic-cae` v0.1.0

## Key Dependencies

**Critical (from `pyproject.toml`):**
- `torch>=2.0.0` - Core ML framework
- `numpy>=1.24.0,<2.0.0` - Array operations
- `matplotlib>=3.7.0` - Visualization
- `scipy>=1.10.0` - Scientific computing, .mat file handling
- `tqdm>=4.65.0` - Progress bars
- `pyyaml` - Config file parsing

**Dev Dependencies:**
- `ipython>=8.0.0` - Interactive Python shell

## Project Structure

```
ultrasonic-cae/
├── model/model.py          # Model definitions (DeepCAE, DeepCAE_PINN, DeepSetsPINN)
├── data/                  # Dataset utilities and dataloaders
│   ├── data_utils.py      # UltrasonicDataset, create_dataloaders
│   └── data_deepsets.py   # DeepSets-specific dataset builder
├── scripts/
│   ├── train/train.py     # Unified training entry point
│   ├── transformer.py     # .mat to train/val directory conversion
│   ├── analysis/          # Inference and validation scripts
│   └── run_unified_pipeline.py  # End-to-end pipeline orchestrator
├── configs/               # JSON pipeline configuration templates
└── results/               # Training outputs (checkpoints, images)
```

## Device Support

Priority order (from `scripts/train/train.py`):
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU)
3. CPU (fallback)

## Configuration

**Pipeline Configs:**
- `configs/pipeline_pinn_template.json` - PINN pipeline parameters
- `configs/pipeline_deepsets_template.json` - DeepSets-PINN pipeline parameters

**Physics Constants (hardcoded in `model/model.py`):**
- Sampling Rate: 6.25 MHz
- Duration: 160 us
- Points: 1000
- Center Frequency: 250 kHz
- Wave Speed: 5900 m/s (steel)

---

*Stack analysis: 2026-04-12*
