# Technology Stack

**Analysis Date:** 2026-04-24

## Languages

**Primary:**
- Python 3.10+ - Core implementation language for all ML and signal processing tasks

## Runtime & Package Manager

**Environment:**
- Python 3.10+ (requires >=3.10 per `pyproject.toml`)
- Virtual environment: `.venv/` (managed via `uv`)

**Package Manager:**
- `uv` - Primary package manager (per CLAUDE.md setup instructions)
- Lockfile: `uv.lock` (present in project root)

## Frameworks & Libraries

**Deep Learning:**
- PyTorch >=2.0.0 - Neural network framework (`torch>=2.0.0`)
- No explicit CUDA version pinned; device priority: cuda > mps > cpu (per CLAUDE.md)

**Scientific Computing:**
- NumPy 1.24.0-2.0.0 - Array operations (`numpy>=1.24.0,<2.0.0`)
- SciPy >=1.10.0 - Signal processing, interpolation (`scipy>=1.10.0`)
- PyWavelets >=1.5.0 - Wavelet transforms (`PyWavelets>=1.5.0`)

**Visualization:**
- Matplotlib >=3.7.0 - Plotting and visualization (`matplotlib>=3.7.0`)

**GUI:**
- PySide6 >=6.6.0 - Qt-based GUI framework (`PySide6>=6.6.0`) (used for train_gui.py and rms_imaging_gui.py)

**Data & Config:**
- PyYAML - YAML config parsing (`pyyaml`)
- tqdm >=4.65.0 - Progress bars (`tqdm>=4.65.0`)

**Build:**
- hatchling - Build backend (`hatchling.build`)

**Dev Dependencies:**
- ipython >=8.0.0 - Interactive Python shell

## Project Structure

**Key Source Directories:**
- `model/` - Neural network model definitions (`DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN`)
- `scripts/` - Training, inference, data transformation scripts
  - `scripts/train/` - Training pipeline (`train.py`, `train_gui.py`)
  - `scripts/analysis/` - Inference and validation (`inference.py`, `inference_deepsets.py`, `acoustic_validation.py`)
  - `scripts/matlab/` - MATLAB integration for imaging
- `data/` - Data directory (train/val structure populated by transformer.py)
- `configs/` - JSON pipeline configuration templates
- `results/` - Training outputs (checkpoints/, images/)

## GPU/Acceleration

**Device Priority:**
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU)
3. CPU (fallback)

**Implementation:** `torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')`

## Physics Constants

Defined in `model/model.py`:
- Sampling Rate: 6.25 MHz
- Duration: 160 us
- Points: 1000
- Center Frequency: 250 kHz
- Wave Speed: 5900 m/s (steel)
- Grid Spacing: 1e-3 m (DX, DY)

## Configuration

**Pipeline Configuration:**
- JSON templates in `configs/` directory
- Key configs: `pipeline_pinn_template.json`, `pipeline_deepsets_template.json`, `pipeline_tf_fusion_template.json`

**Environment Variables:**
- Not extensively used; configuration passed via JSON configs and CLI arguments

---

*Stack analysis: 2026-04-24*
