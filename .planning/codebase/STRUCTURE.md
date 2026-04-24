# Codebase Structure

**Analysis Date:** 2026-04-24

## Directory Layout

```
/Users/zyt/ANW/DnCNN2/
├── model/                     # Neural network model definitions
│   ├── __init__.py
│   └── model.py               # DeepCAE, DeepCAE_PINN, DeepSetsPINN, DeepSetsPINN_TF
├── data/                      # Dataset utilities and loaders
│   ├── __init__.py
│   ├── data_utils.py          # UltrasonicDataset, create_dataloaders
│   ├── data_deepsets.py       # DeepSetsDataset, create_deepsets_dataloaders
│   └── tf_features.py         # STFT time-frequency feature extraction
├── scripts/                   # Main scripts
│   ├── train/                 # Training scripts
│   │   ├── train.py           # Unified trainer (pinn + deepsets)
│   │   ├── train_gui.py       # GUI for training
│   │   └── visualization.py   # Plotting utilities
│   ├── analysis/              # Inference and validation
│   │   ├── inference.py       # PINN inference script
│   │   ├── inference_deepsets.py  # DeepSets inference script
│   │   ├── acoustic_validation.py # Acoustic feature validation
│   │   └── preview_signals.py # Signal preview utility
│   ├── matlab/                # MATLAB utilities
│   │   ├── imaging/           # RMS imaging
│   │   ├── analysis/          # Analysis scripts
│   │   ├── filtering/          # Filtering utilities
│   │   ├── dispersion/         # Dispersion analysis
│   │   └── io/                # I/O utilities
│   ├── transformer.py         # Data transformation (.mat -> dataset)
│   ├── run_unified_pipeline.py # End-to-end pipeline orchestrator
│   └── git_auto.py           # Git automation
├── configs/                   # Pipeline configuration files
│   ├── pipeline_pinn.json
│   ├── pipeline_pinn_template.json
│   ├── pipeline_deepsets.json
│   ├── pipeline_deepsets_template.json
│   ├── pipeline_tf_fusion.json
│   ├── pipeline_tf_fusion_template.json
│   ├── pipeline_tf_fusion_v2.json
│   ├── pipeline_tf_fusion_remote.json
│   ├── ablation_exp_A_baseline.json
│   ├── ablation_exp_B_concat_fusion.json
│   └── ablation_exp_C_low_physics.json
├── data/                      # Training/validation data (created by transformer)
│   ├── train/
│   │   ├── clean/
│   │   ├── noisy/
│   │   └── tf/               # Time-frequency features (for tf_fusion)
│   ├── val/
│   │   ├── clean/
│   │   ├── noisy/
│   │   └── tf/
│   └── metadata.npy
├── results/                   # Training outputs (timestamped runs)
│   ├── <timestamp>/
│   │   ├── checkpoints/
│   │   ├── images/
│   │   └── experiments/      # Markdown experiment records
│   └── images/               # Aggregated images
├── checkpoints/              # Legacy/checkpoint storage
├── archive/                  # Archived experiments
├── docs/                     # Documentation
│   └── superpowers/          # Design specs and plans
├── pyproject.toml            # Project dependencies
└── CLAUDE.md                 # Project instructions

```

## Directory Purposes

**model/:**
- Purpose: Neural network architecture definitions
- Contains: All model classes (DeepCAE, DeepCAE_PINN, DeepSetsPINN, DeepSetsPINN_TF, encoders, decoders, fusion modules)
- Key file: `model/model.py` (all model definitions)

**data/:**
- Purpose: Dataset loading and preprocessing utilities
- Contains: PyTorch Dataset classes, dataloader factories, time-frequency feature extraction
- Key files: `data_utils.py`, `data_deepsets.py`, `tf_features.py`

**scripts/:**
- Purpose: Executable scripts for training, inference, and data transformation
- Contains: train/, analysis/, matlab/ subdirectories
- Key files: `train/train.py`, `analysis/inference.py`, `analysis/inference_deepsets.py`, `transformer.py`, `run_unified_pipeline.py`

**scripts/train/:**
- Purpose: Training-related scripts
- Contains: `train.py` (main trainer), `train_gui.py` (optional GUI), `visualization.py` (plotting)

**scripts/analysis/:**
- Purpose: Inference and validation scripts
- Contains: `inference.py` (PINN), `inference_deepsets.py` (DeepSets), `acoustic_validation.py`, `preview_signals.py`

**scripts/matlab/:**
- Purpose: MATLAB utilities for imaging and signal processing
- Contains: `imaging/` (RMS imaging), `analysis/`, `filtering/`, `dispersion/`, `io/`

**configs/:**
- Purpose: JSON configuration files for pipeline parameters
- Key files: `pipeline_pinn_template.json`, `pipeline_deepsets_template.json`, `pipeline_tf_fusion_template.json`

**data/ (data directory):**
- Purpose: Training and validation data storage (created by transformer.py)
- Structure: `train/{clean,noisy,tf}/` and `val/{clean,noisy,tf}/` with .npy files

**results/:**
- Purpose: Training outputs organized by timestamp
- Structure: `<timestamp>/{checkpoints,images,experiments}/`

## Key File Locations

**Entry Points:**
- `scripts/run_unified_pipeline.py`: Unified end-to-end pipeline (transform + train + inference)
- `scripts/train/train.py`: Training entry point (called by unified pipeline)
- `scripts/transformer.py`: Data transformation entry point

**Configuration:**
- `configs/pipeline_pinn_template.json`: Template for PINN pipeline
- `configs/pipeline_deepsets_template.json`: Template for DeepSets pipeline
- `configs/pipeline_tf_fusion_template.json`: Template for tf_fusion pipeline

**Core Logic:**
- `model/model.py`: All neural network model definitions
- `data/data_utils.py`: UltrasonicDataset and dataloaders for PINN
- `data/data_deepsets.py`: DeepSetsDataset and dataloaders for DeepSets
- `data/tf_features.py`: STFT-based time-frequency feature extraction

**Inference:**
- `scripts/analysis/inference.py`: PINN model inference
- `scripts/analysis/inference_deepsets.py`: DeepSets model inference

**Testing:**
- Not applicable (no formal test suite detected)

## Naming Conventions

**Files:**
- Python modules: `snake_case.py`
- Config files: `snake_case.json`
- Data directories: `train/`, `val/`
- Output directories: `<timestamp>/` format (e.g., `20260421_231648/`)

**Functions/Classes:**
- Models: `PascalCase` (e.g., `DeepCAE`, `DeepSetsPINN`)
- Dataset classes: `PascalCase` (e.g., `UltrasonicDataset`, `DeepSetsDataset`)
- Utility functions: `snake_case` (e.g., `create_dataloaders`, `load_mat_file`)
- Training functions: `snake_case` (e.g., `train_pinn`, `train_deepsets_pinn`)

**Constants:**
- Module-level constants: `UPPER_SNAKE_CASE` (e.g., `SAMPLING_RATE`, `DURATION`)
- Physics constants defined in `model/model.py` lines 12-18

## Where to Add New Code

**New Model Variant:**
- Primary code: `model/model.py` (add new class extending existing architecture)
- Training: `scripts/train/train.py` (add new train function or extend `train_from_config`)
- Inference: `scripts/analysis/inference.py` or `scripts/analysis/inference_deepsets.py`

**New Data Processing:**
- Dataset class: `data/data_utils.py` or `data/data_deepsets.py`
- TF features: `data/tf_features.py`
- Transformation: `scripts/transformer.py`

**New Analysis/Validation:**
- Scripts: `scripts/analysis/`
- Visualization: `scripts/train/visualization.py`

**New Configuration:**
- Add JSON to `configs/` following `*_template.json` pattern

## Special Directories

**results/:**
- Purpose: All training outputs (checkpoints, images, experiment records)
- Generated: Yes (created during training runs)
- Committed: No (typically in .gitignore)

**data/ (data directory):**
- Purpose: Transformed training/validation data
- Generated: Yes (by `transformer.py`)
- Committed: No (typically in .gitignore)

**.venv/:**
- Purpose: Python virtual environment
- Generated: Yes (by `uv sync`)
- Committed: No

**.worktrees/:**
- Purpose: Git worktrees for parallel development
- Generated: Yes (by git worktree commands)
- Committed: No

---

*Structure analysis: 2026-04-24*
