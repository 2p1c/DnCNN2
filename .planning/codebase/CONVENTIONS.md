# Coding Conventions

**Analysis Date:** 2026-04-24

## Python Version and Environment

**Version:** Python >= 3.10
**Package Manager:** `uv` (managed via `pyproject.toml`)
**Execution Pattern:** Scripts run via `uv run python <script>.py`

## Import Patterns

**Future imports:**
```python
from __future__ import annotations
```
Used in all Python files for forward references and type annotations.

**Script imports (non-package scripts):**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
```
Scripts in `scripts/` add the repo root to `sys.path` to import from `model/`, `data/`, etc.

**Package imports:**
```python
from model.model import DeepCAE, DeepCAE_PINN
from data import create_dataloaders
```

## Naming Conventions

**Files:** `snake_case.py`

**Classes:** `PascalCase`
- `DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN`, `DeepSetsPINN_TF`
- `SignalEncoder`, `CoordinateMLP`, `PointEncoder`
- `PINNLoss`, `DeepSetsPINNLoss`

**Functions:** `snake_case`
- `train_epoch_pinn`, `validate_pinn`, `run_inference`
- `calculate_psnr`, `calculate_snr`
- `load_model`, `preprocess_mat_data`, `denoise_signals`

**Constants:** `UPPER_SNAKE_CASE`
```python
SAMPLING_RATE: float = 6.25e6
DURATION: float = 160e-6
NUM_POINTS: int = 1000
CENTER_FREQUENCY: float = 250e3
DEFAULT_WAVE_SPEED: float = 5900.0
```

**Type Variables:** Full names or conventional single letters
```python
from typing import Dict, Tuple, Optional
```

## Code Style

**Formatter:** Not explicitly configured (AGENTS.md notes no lint/typecheck is defined)

**Key Style Rules Observed:**
- Indentation: 4 spaces
- Line length: often exceeds 100 chars in long argument lists or log statements
- `inplace=True` for activation functions where applicable
- `torch.no_grad()` context manager for inference
- Type annotations on function signatures

## File Organization

**Directory Structure:**
```
/Users/zyt/ANW/DnCNN2/
├── model/
│   └── model.py           # All model definitions (DeepCAE, DeepCAE_PINN, DeepSetsPINN, etc.)
├── scripts/
│   ├── analysis/          # Inference and validation scripts
│   │   ├── inference.py
│   │   ├── inference_deepsets.py
│   │   ├── acoustic_validation.py
│   │   └── preview_signals.py
│   ├── train/
│   │   └── train.py       # Unified training entry point
│   ├── transformer.py     # Data preparation (.mat -> directory structure)
│   └── run_unified_pipeline.py  # End-to-end orchestrator
├── data/                 # Data loading and processing
│   ├── __init__.py
│   ├── data_utils.py
│   ├── data_deepsets.py
│   └── tf_features.py
├── configs/               # JSON pipeline templates
│   ├── pipeline_pinn_template.json
│   ├── pipeline_deepsets_template.json
│   ├── pipeline_tf_fusion_template.json
│   └── ablation_exp_*.json
├── results/               # Output directory (gitignored)
├── checkpoints/          # Model weights (gitignored)
└── docs/                  # Documentation
```

## Commit Message Style

**Format:** Conventional Commits
```
<type>: <description>
```

**Types observed in git log:**
- `feat`: New features
- `fix`: Bug fixes
- `refactor`: Code refactoring
- `docs`: Documentation
- `test`: Tests
- `chore`: Maintenance
- `perf`: Performance

## Configuration Format

**JSON Config Templates:**
Located in `configs/`:
- `pipeline_pinn_template.json`
- `pipeline_deepsets_template.json`
- `pipeline_tf_fusion_template.json`
- `ablation_exp_*.json` for ablation studies

**Key config parameters:**
```json
{
  "pipeline": "pinn|deepsets",
  "data_dir": "data",
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001,
  "physics_weight": 0.001,
  "wave_speed": 5900.0,
  "center_frequency": 250000.0,
  "signal_length": 1000
}
```

## Key Physics Constants

| Constant | Value | Location |
|----------|-------|----------|
| Sampling Rate | 6.25 MHz | `model/model.py`, `scripts/analysis/acoustic_validation.py` |
| Duration | 160 us | `model/model.py` |
| Points | 1000 | `model/model.py` |
| Center Frequency | 250 kHz | `model/model.py` |
| Wave Speed | 5900 m/s | `model/model.py` |

## Error Handling Patterns

**Explicit validation with clear messages:**
```python
if signal_data.shape[1] != target_signal_length:
    raise ValueError(
        f"Signal length mismatch after preprocessing: "
        f"got {signal_data.shape[1]}, expected {target_signal_length}"
    )
```

**Device selection with fallback:**
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

## Logging Patterns

**Console output with tags:**
```python
print(f"[INFO] Loading model from {checkpoint_path}")
print("[WARNING] Denormalized output is all zeros...")
print("[TRAIN_METRIC] pipeline=pinn epoch={epoch} ...")
```

**Progressive processing logs:**
```python
print(f"\n[Step 1] Loading .mat file: {mat_path}")
print(f"\n[Step 2] Reshaping to {grid_cols}x{grid_rows} grid...")
```

## Documentation

**Docstrings:** Present in key functions
**Inline comments:** Used for physics-related code (e.g., `# 物理常数`)

---

*Convention analysis: 2026-04-24*
