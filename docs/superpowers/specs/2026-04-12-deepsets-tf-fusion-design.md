# DeepSets TF-Fusion Design

## Context

This repo currently keeps two baseline model families for comparison experiments:

- `DeepCAE` / `DeepCAE_PINN` (1D CAE + physics residual)
- `DeepSetsPINN` (set-invariant patch model + 2D wave-equation residual)

Goal: add a **new** model variant without modifying existing two baseline paths, so old models remain available for ablation.

## Goal

Add a new DeepSets-based model that inherits from the wave-equation embedded model and introduces an extra time-frequency information channel.

Selected direction:

- dual-branch architecture (time-domain branch + time-frequency branch)
- offline STFT log-magnitude feature generation
- gated feature fusion before `PointEncoder`

## Scope

In scope:

- New model class `DeepSetsPINN_TF` inheriting from `DeepSetsPINN`
- New modules: `TimeFreqEncoder`, `GatedFeatureFusion`
- Data path support for offline TF signals (`train/tf`, `val/tf`)
- Training/inference config switch via `model_type=tf_fusion`
- Checkpoint metadata and strict parameter consistency checks
- Ablation-ready experiment protocol

Out of scope:

- Replacing or modifying behavior of `DeepCAE`, `DeepCAE_PINN`, `DeepSetsPINN`
- Changing wave-equation residual definition
- Migrating pipelines away from current script-driven entrypoints

## Design Options Considered

### Option A (Chosen): Dual-Branch + Gated Fusion + Offline STFT

- Time branch uses existing `SignalEncoder`
- TF branch uses new `TimeFreqEncoder`
- Fusion uses learned gate `g` to mix branch embeddings
- Reuses existing `PointEncoder`, set aggregation, decoder, and physics residual

Why chosen:

- better capacity and interpretability than single-encoder 2-channel input
- moderate implementation cost
- clean ablation against existing `DeepSetsPINN`

### Option B: Dual-Branch + Concat Fusion

- same branch split as Option A
- fusion by concat + MLP only

Trade-off: simpler but less adaptive under sample-dependent noise characteristics.

### Option C: Two-Channel Single Encoder

- stack time and TF sequences into 2-channel 1D input
- one shared encoder

Trade-off: lightest implementation, but weaker branch disentanglement and lower interpretability.

## Target Architecture

### New Modules

1) `TimeFreqEncoder(nn.Module)`

- input shape: `[B*R, 1, T]`
- output shape: `[B*R, signal_embed_dim]`
- architecture aligned with `SignalEncoder` depth for stable feature scale

2) `GatedFeatureFusion(nn.Module)`

- inputs: `f_time`, `f_tf` in `R^D`
- gate: `g = sigmoid(MLP([f_time; f_tf]))`
- output: `f_fused = g * f_time + (1 - g) * f_tf`
- recommended post-op: `LayerNorm(D)`

3) `DeepSetsPINN_TF(DeepSetsPINN)`

- calls `super().__init__(...)` to preserve all existing DeepSets physics wiring
- adds `self.tf_encoder` and `self.fusion`
- overrides `forward` and `physics_forward` signatures to accept `tf_signals`

### Forward Path

Given:

- `noisy_signals`: `[B, R, T]`
- `tf_signals`: `[B, R, T]`
- `coordinates`: `[B, R, 2]`

Flow:

1. `sig_emb = signal_encoder(noisy_signals)`
2. `tf_emb = tf_encoder(tf_signals)`
3. `fused_emb = fusion(sig_emb, tf_emb)`
4. `coord_emb = coord_mlp(coordinates)`
5. `point_feat = point_encoder(fused_emb, coord_emb)`
6. `global_feat = mean(point_feat, dim=1)`
7. `decoder(cat(point_feat, global_feat)) -> denoised`

Physics path:

- reuse `compute_wave_equation_residual` from `DeepSetsPINN`
- `physics_forward` returns `(denoised, residual)` exactly like current interface style

## Data and Feature Pipeline

### TF Feature Source (Offline)

For each 1D signal:

1. STFT on original time-domain signal
2. magnitude map: `|S|`
3. log compression: `log(1 + |S|)`
4. frequency-axis pooling to a 1D temporal sequence
5. resample/interpolate to length `T=1000`
6. per-signal normalization to `[-1, 1]`

Output directory layout:

- `data/train/tf`
- `data/val/tf`

File indexing/naming must align with existing noisy samples one-to-one.

### Dataloader Contract

Existing DeepSets batch keys remain. New key for TF branch:

- `tf_signals`: `[B, R, T]`

New model path must fail fast when TF files are missing or count mismatches noisy files.

## Training and Inference Integration

### Training

- Keep `pipeline=deepsets`
- Add model switch: `model_type in {deepsets, tf_fusion}`
- `deepsets`: existing model path unchanged
- `tf_fusion`: instantiate `DeepSetsPINN_TF`, read `tf_signals` from batch, call new `physics_forward`

Loss remains unchanged:

- `total = data_loss + physics_weight * physics_loss`

### Inference

- Extend DeepSets inference path to support `model_type=tf_fusion`
- Build TF features using same STFT parameter set used in training
- enforce strict parameter consistency between runtime args and checkpoint metadata

## Config and Checkpoint Schema

### New Config Fields

- `model_type`: `deepsets` or `tf_fusion`
- STFT params: `n_fft`, `hop_length`, `win_length`, `stft_window`, `tf_pooling`
- `tf_embed_dim` (default equal to `signal_embed_dim`)

### Checkpoint Metadata Additions

- `model_type`
- full STFT parameter set
- TF embed dimensions

Inference loader must hard-fail on incompatible checkpoint/model_type or STFT settings.

## Error Handling Requirements

- Missing `train/tf` or `val/tf` directories -> clear `FileNotFoundError`
- Sample count mismatch among `noisy/clean/tf` -> clear `ValueError`
- Inference STFT config mismatch with checkpoint -> clear `ValueError`
- Any NaN/Inf in branch embedding, fusion output, or residual -> explicit runtime check in debug mode

## Ablation Plan

Required comparisons:

- A0: `DeepSetsPINN` baseline
- A1: `DeepSetsPINN_TF` + concat fusion
- A2: `DeepSetsPINN_TF` + gated fusion (main)
- A3: A2 with physics weight sweep

Report metrics:

- PSNR
- wave-equation residual statistics
- existing acoustic validation metrics
- parameter count and training wall-clock

## Validation and Acceptance Criteria

### Functional

- New model trains end-to-end in file mode without touching old model classes
- New inference path outputs `.mat` and validation plots successfully
- Existing `pinn` and `deepsets` commands still run unchanged

### Numerical

- no NaN/Inf in training and inference for smoke runs
- gate values remain in `[0, 1]`

### Experiment

- A2 must match or improve at least one primary quality metric over A0
- no large regression in physics residual quality

## Risks and Mitigations

1) TF branch dominates and over-smooths outputs

- mitigation: gate monitoring, smaller TF encoder capacity, fusion dropout

2) Offline/online STFT inconsistency

- mitigation: checkpointed STFT params + strict inference validation

3) Runtime overhead increase

- mitigation: offline TF caching, tuned TF encoder width

## Rollout Strategy

1. Add modules and model class behind `model_type=tf_fusion`
2. Add TF data pipeline and dataloader checks
3. Add train/inference integration with backward compatibility
4. Run smoke checks
5. Run ablations and finalize default settings

## Non-Regression Requirement

The following existing paths are mandatory to preserve:

- `scripts/train/train.py --pipeline pinn ...`
- `scripts/train/train.py --pipeline deepsets ...` (baseline mode)
- `scripts/run_unified_pipeline.py` for baseline `pinn` and `deepsets`

No behavior changes are allowed for these baseline paths.
