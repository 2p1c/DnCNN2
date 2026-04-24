# Testing Patterns

**Analysis Date:** 2026-04-24

## Testing Framework

**Framework:** No formal test framework (pytest) configured
**Status:** Per `AGENTS.md`: "No lint/typecheck/pytest config is defined; use script-level validation"

**Verification instead of testing:**
- Script-level validation through executable smoke tests
- Acoustic feature validation through `acoustic_validation.py`
- Signal preview through `preview_signals.py`

## Validation Scripts

### Acoustic Validation (`scripts/analysis/acoustic_validation.py`)

**Purpose:** Comprehensive acoustic feature validation for training and inference

**Training Mode:**
```python
run_acoustic_validation(model, val_loader, device, save_path, num_samples)
```

**Inference Mode:**
```python
run_inference_validation(input_signals, denoised_signals, save_path, num_samples)
```

**Features Extracted:**
- Time domain: arrival time, peak amplitude, RMS, crest factor, zero-crossing rate
- Frequency domain: spectral centroid, bandwidth, dominant frequency, -3dB bandwidth
- Energy: total energy, sub-band energy distribution (0-100k, 100k-200k, 200k-400k, >400k)
- Wavenumber: phase linearity, dominant wavenumber, spectral energy concentration
- Quality metrics: cross-correlation, spectral coherence, envelope correlation

**Usage (standalone):**
```bash
uv run python scripts/analysis/acoustic_validation.py
```

### Preview Signals (`scripts/analysis/preview_signals.py`)

**Purpose:** Visual preview of signals with detailed analysis

**Usage:**
```bash
uv run python scripts/analysis/preview_signals.py --detailed --no_show
```

### Inference (`scripts/analysis/inference.py`)

**Purpose:** Model inference on .mat files with acoustic validation

**Usage:**
```bash
uv run python scripts/analysis/inference.py --input data/noisy.mat --output results/ --checkpoint <path>
```

## Smoke Tests

**Defined in `AGENTS.md`:**

**PINN synthetic smoke:**
```bash
uv run python scripts/train/train.py --pipeline pinn --epochs 1 --num_train 64 --num_val 16
```

**PINN file-mode smoke:**
```bash
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --epochs 1
```

**DeepSets file-mode smoke:**
```bash
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 1 --batch_size 8
```

**Single inference smoke:**
```bash
uv run python scripts/analysis/inference.py --input data/noisy.mat --output results/
```

**End-to-end skip smoke:**
```bash
uv run python scripts/run_unified_pipeline.py --pipeline pinn --inference_input data/noisy.mat --skip_transform --skip_train --checkpoint results/checkpoints/best_pinn_model.pth
```

## Verification Commands

**From `CLAUDE.md`:**
```bash
uv run python -m compileall .
uv run python scripts/analysis/acoustic_validation.py
uv run python scripts/analysis/preview_signals.py --detailed --no_show
uv build
```

**Execution pattern:** All scripts run via `uv run python <script>.py`

## CI Configuration

**File:** `.github/workflows/ci.yml`

**Purpose:** MkDocs documentation deployment (NOT a test/lint gate)
```yaml
name: ci
on:
  push:
    branches:
      - develop
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install mkdocs-material
      - run: mkdocs gh-deploy --force
```

**Note:** CI does NOT run training, inference, or validation tests. It only deploys documentation.

## Coverage

**Enforcement:** None
**Approach:** Script-level validation and smoke tests instead of unit test coverage

## Test Organization

**Location:** No formal test directory
**Pattern:** Tests are embedded in validation scripts or run as standalone smoke tests

**Data for testing:**
- Synthetic data generated on-the-fly via `data/data_utils.py`
- File-based data from `.mat` files in `data/` directory

## Model Validation Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
```python
def calculate_psnr(clean, denoised, max_val=1.0):
    mse = torch.mean((clean - denoised) ** 2).item()
    return 10 * np.log10(max_val**2 / mse)
```

**SNR (Signal-to-Noise Ratio):**
```python
def calculate_snr(signal, noise):
    signal_power = torch.mean(signal**2).item()
    noise_power = torch.mean(noise**2).item()
    return 10 * np.log10(signal_power / noise_power)
```

**Acoustic Quality Thresholds:**
- Good cross-correlation: >= 0.9
- Fair cross-correlation: >= 0.7
- Good spectral coherence (100-500kHz): >= 0.8
- Good arrival time error: <= 2.0 us
- Feature preservation target: >= 80%

## No Formal Test Files

**Note:** There are no `test_*.py` or `*_test.py` files in this codebase. All validation is performed through:
1. Executable scripts (`scripts/analysis/`)
2. Smoke tests (documented in `AGENTS.md`)
3. Build verification (`uv build`)

---

*Testing analysis: 2026-04-24*
