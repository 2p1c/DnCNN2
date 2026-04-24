# Architecture

**Analysis Date:** 2026-04-24

## Pattern Overview

**Overall:** Physics-informed neural network (PINN) for ultrasonic signal denoising

**Key Characteristics:**
- Two independent pipelines: `pinn` (DeepCAE) and `deepsets` (DeepSets)
- Physics constraints enforced via wave equation residuals
- Set-invariant architecture for spatial grid data
- Optional time-frequency fusion branch for enhanced denoising

## Layers

**Model Definition:**
- Purpose: Define neural network architectures for signal denoising
- Location: `model/model.py`
- Contains: `DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN`, `DeepSetsPINN_TF`, encoders, decoders, fusion modules
- Depends on: PyTorch (torch.nn)

**Data Loading:**
- Purpose: Load and prepare training/inference data
- Location: `data/data_utils.py`, `data/data_deepsets.py`, `data/tf_features.py`
- Contains: `UltrasonicDataset`, `DeepSetsDataset`, dataloader factories
- Depends on: NumPy, PyTorch DataLoader

**Training:**
- Purpose: Train models with physics-informed loss functions
- Location: `scripts/train/train.py`
- Contains: `train_pinn()`, `train_deepsets_pinn()`, loss classes, training loops
- Depends on: model, data utilities, visualization

**Inference:**
- Purpose: Apply trained models to new data
- Location: `scripts/analysis/inference.py`, `scripts/analysis/inference_deepsets.py`
- Contains: Model loading, preprocessing, denoising, denormalization, saving
- Depends on: model, transformer utilities, acoustic validation

## Data Flow

**Pipeline: pinn**

1. Load noisy (21x21) and clean (41x41) .mat files via `scripts/transformer.py`
2. Interpolate noisy grid to match clean grid resolution (cubic/linear)
3. Augment data (time_shift, amplitude_scale, flip, add_noise, time_stretch, window_crop)
4. Split into train/val sets, save as .npy files
5. Load data via `UltrasonicDataset` in `data/data_utils.py`
6. Train `DeepCAE_PINN` with combined MSE + physics residual loss
7. Run inference via `scripts/analysis/inference.py`

**Pipeline: deepsets**

1. Same transformer steps as pinn
2. Additionally compute STFT-based time-frequency features (when `model_type=tf_fusion`)
3. Load data via `DeepSetsDataset` which organizes 41x41 grid as patches with coordinates
4. Train `DeepSetsPINN` or `DeepSetsPINN_TF` with 2D wave equation residual
5. Run inference via `scripts/analysis/inference_deepsets.py`

**Unified Pipeline:**
- Entry: `scripts/run_unified_pipeline.py`
- Orchestrates: transform -> train -> inference in one command
- Supports both `pinn` and `deepsets` pipelines

## Key Abstractions

**DeepCAE (Convolutional Autoencoder):**
- Purpose: Baseline 1D signal denoising autoencoder
- Examples: `model/model.py` lines 21-156
- Pattern: 5-layer encoder with stride-2 convs, 5-layer decoder with transposed convs

**DeepCAE_PINN:**
- Purpose: Physics-informed DeepCAE with damped narrowband residual
- Examples: `model/model.py` lines 159-210
- Pattern: Extends DeepCAE, adds `physics_forward()` computing wave equation residual
- Physics: Damped harmonic oscillator equation at 250 kHz center frequency

**DeepSetsPINN:**
- Purpose: Set-invariant model operating on spatial grid patches
- Examples: `model/model.py` lines 446-571
- Pattern: SignalEncoder -> PointEncoder with coordinates -> Decoder
- Physics: 2D wave equation residual computed over patch neighborhood

**DeepSetsPINN_TF:**
- Purpose: DeepSetsPINN with additional time-frequency fusion branch
- Examples: `model/model.py` lines 574-675
- Pattern: Dual encoder (time + time-frequency) with GatedFeatureFusion or ConcatFeatureFusion
- Fusion modes: `gated` (adaptive weighted blend) or `concat` (projected concatenation)

**SignalEncoder / TimeFreqEncoder:**
- Purpose: Encode 1D signals into embedding space
- Examples: `model/model.py` lines 213-282
- Pattern: 5-layer 1D CNN with adaptive average pooling + linear projection

**GatedFeatureFusion / ConcatFeatureFusion:**
- Purpose: Combine time and time-frequency embeddings
- Examples: `model/model.py` lines 316-357
- Pattern: GatedFeatureFusion uses sigmoid gate to adaptively weight f_time vs f_tf

## Entry Points

**Training:**
- Location: `scripts/train/train.py`
- Triggers: Direct script execution or called by `run_unified_pipeline.py`
- Responsibilities: Model initialization, training loop, checkpointing, visualization

**Unified Pipeline:**
- Location: `scripts/run_unified_pipeline.py`
- Triggers: `uv run python scripts/run_unified_pipeline.py --config configs/...`
- Responsibilities: Orchestrate all stages (transform, train, inference)

**Data Transformation:**
- Location: `scripts/transformer.py`
- Triggers: Called by unified pipeline or direct execution
- Responsibilities: Load .mat files, interpolate, augment, save dataset

**Inference:**
- Location: `scripts/analysis/inference.py` (pinn), `scripts/analysis/inference_deepsets.py` (deepsets)
- Triggers: Called by unified pipeline or direct execution
- Responsibilities: Load checkpoint, preprocess, denoise, save results

## Error Handling

**Strategy:** Exceptions with informative messages, graceful degradation where possible

**Patterns:**
- Checkpoint loading validates model_type mismatch
- Interpolation handles identical source/target sizes as no-op
- Tiny signal amplitude triggers automatic linear interpolation fallback
- Edge points with no patch prediction fall back to input signal

## Cross-Cutting Concerns

**Logging:** Print statements with `[INFO]`, `[Step N]` prefixes; training metrics logged with `[TRAIN_METRIC]` prefix

**Validation:** Acoustic validation runs after training (arrival time, peak amplitude, RMS, frequency preservation)

**Device Priority:** CUDA > MPS > CPU (automatic selection in `_select_device()`)

**Physics Constants:**
- Sampling Rate: 6.25 MHz
- Duration: 160 us
- Points: 1000
- Center Frequency: 250 kHz
- Wave Speed: 5900 m/s (steel)
- Grid Spacing: 1 mm

---

*Architecture analysis: 2026-04-24*
