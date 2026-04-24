# External Integrations

**Analysis Date:** 2026-04-24

## Data File Formats

**Input (.mat files):**
- MATLAB binary format loaded via `scipy.io.loadmat()`
- Noisy signals: 21x21 grid stored as `data/noisy.mat`
- Clean signals: 41x41 grid stored as `data/clean.mat`
- Loaded by `scripts/transformer.py` using `scipy.io.loadmat(filepath)`

**Input Variables in .mat files:**
- `x` - Time vector (extracted via `mat_data["x"].flatten()`)
- `y` - Signal data array of shape (n_points, signal_length)

**Output Data Format:**
- NumPy `.npy` files saved by `scripts/transformer.py`
- Directory structure: `data/{train,val}/{noisy,clean}/0000.npy`
- Metadata saved to `data/metadata.npy`

## External Services

**No external API integrations detected:**
- No cloud storage services
- No external ML model APIs
- No remote data sources

**Training is fully local:**
- Models train entirely on local hardware
- Checkpoints saved to local `results/<timestamp>/checkpoints/` directory

## Hardware Integrations

**Ultrasonic NDT Scanning System (indirect):**
- Input .mat files originate from ultrasonic scanning equipment
- Grid sizes (21x21 noisy, 41x41 clean) reflect scanning probe configuration
- Scanning parameters embedded in physics constants:
  - Wave Speed: 5900 m/s (steel)
  - Center Frequency: 250 kHz
  - Sampling Rate: 6.25 MHz
  - Grid Spacing: 1e-3 m

**GPU Acceleration:**
- CUDA (NVIDIA) - preferred if available
- MPS (Apple Silicon M-series) - fallback on Mac
- CPU - final fallback

## Internal Pipeline Integrations

**Data Transform Pipeline:**
```
scripts/transformer.py
  - Input: data/noisy.mat, data/clean.mat
  - Output: data/train/{noisy,clean}/, data/val/{noisy,clean}/
  - Optional TF features: data/{train,val}/tf/
```

**Training Pipeline:**
```
scripts/train/train.py
  - Input: data/ directory (from transformer)
  - Config: configs/pipeline_*.json
  - Output: results/<timestamp>/checkpoints/
```

**Inference Pipeline:**
```
scripts/analysis/inference.py (PINN)
scripts/analysis/inference_deepsets.py (DeepSets)
  - Input: data/*.mat or processed signals
  - Checkpoint: results/<timestamp>/checkpoints/best_model.pth
  - Output: results/<timestamp>/images/
```

**Unified Pipeline:**
```
scripts/run_unified_pipeline.py
  - Orchestrates: transformer -> train -> inference
  - Config: configs/pipeline_*.json
```

## Configuration Files

**Pipeline Configs (JSON):**
- `configs/pipeline_pinn_template.json` - PINN pipeline config
- `configs/pipeline_deepsets_template.json` - DeepSets pipeline config
- `configs/pipeline_tf_fusion_template.json` - TF-fusion pipeline config
- `configs/pipeline_tf_fusion.json` - Active TF-fusion config (modified on branch)

**Architecture Specs (JSON):**
- `tf_fusion_arch.json`, `tf_fusion_architecture.json` - TF fusion architecture definitions
- `gated_fusion_detail.json` - Gated fusion architecture details

## Environment Configuration

**No explicit environment variables required:**
- All configuration via JSON config files and CLI arguments
- Device selection automatic (cuda > mps > cpu)
- Paths are relative or specified at runtime

**Data paths are hardcoded defaults:**
- `data/noisy.mat`, `data/clean.mat` - Default input locations
- `data/` - Default output directory for transformed data
- `results/` - Default directory for training outputs

---

*Integration audit: 2026-04-24*
