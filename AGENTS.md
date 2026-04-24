# AGENTS.md for DnCNN2

## Scope
- Python package: `ultrasonic-cae` (`requires-python >=3.10`), managed with `uv`.
- Active pipelines are only `pinn` and `deepsets` (see `README.md` and `scripts/train/train.py`).
- Main wiring points: `scripts/run_unified_pipeline.py` (orchestration), `scripts/train/train.py` (training), `scripts/analysis/inference.py` and `scripts/analysis/inference_deepsets.py` (inference), `model/model.py` (models).

## Instruction Precedence
- Follow: `CLAUDE.md`, `.github/copilot-instructions.md`, `copilot-instructions.md`.
- Those files include generic web-dev guidance; for this repo prefer the Python/ML workflow and executable scripts.

## Setup and Commands
- Install deps: `uv sync`
- Install dev extras: `uv sync --group dev`
- Run scripts from repo root: `uv run python <script>.py ...`

## High-Value Verification
- No lint/typecheck/pytest config is defined; use script-level validation.
- Recommended checks:
  - `uv run python -m compileall .`
  - `uv run python scripts/analysis/acoustic_validation.py`
  - `uv run python scripts/analysis/preview_signals.py --detailed --no_show`
  - `uv build`

## Focused Sanity Checks
- PINN synthetic smoke: `uv run python scripts/train/train.py --pipeline pinn --epochs 1 --num_train 64 --num_val 16`
- PINN file-mode smoke: `uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --epochs 1`
- DeepSets file-mode smoke: `uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 1 --batch_size 8`
- Single inference smoke: `uv run python scripts/analysis/inference.py --input data/noisy.mat --output results/`
- End-to-end skip smoke:
  - `uv run python scripts/run_unified_pipeline.py --pipeline pinn --inference_input data/noisy.mat --skip_transform --skip_train --checkpoint results/checkpoints/best_pinn_model.pth`

## Workflow Gotchas
- `scripts/run_unified_pipeline.py` requires `--inference_input` even if training is the main goal.
- `--skip_transform` requires prebuilt dataset layout under `data_dir`: `train/noisy`, `train/clean`, `val/noisy`, `val/clean`.
- If `physics_weight` is omitted, defaults are pipeline-specific:
  - `pinn`: `1e-3`
  - `deepsets`: `1e-4`
- `scripts/analysis/inference.py` hard-checks `--signal_length == 1000`.
- Device priority in train/inference is explicit: `cuda` > `mps` > `cpu`.
- Unified pipeline writes checkpoints into `results/<timestamp>/checkpoints` for fresh runs, but when `--skip_train` (and no `--checkpoint`) it looks for checkpoint in root `results/checkpoints/`.
- DeepSets TF-Fusion mode (`--model_type tf_fusion`) requires pre-computed STFT data in `data/train/tf` and `data/val/tf`.

## Artifacts and CI Reality
- Generated data/artifacts are typically untracked (`data/`, `results/`, `*.pth`, `uv.lock` are gitignored).
- `.github/workflows/ci.yml` deploys MkDocs; it is not a test/lint gate for training/inference code.

## Code Conventions To Preserve
- Keep edits minimal/local; avoid unrelated refactors.
- Preserve current script import style (`sys.path.append(...)`) unless intentionally migrating import structure.
- Keep early, explicit validation for file paths and tensor/signal shape assumptions.
- Keep core signal constants aligned unless physics assumptions are intentionally changed:
  - sampling rate `6.25e6`
  - duration `160e-6`
  - points `1000`
  - center frequency `250e3`

## Config Templates
- `configs/pipeline_pinn_template.json` - PINN pipeline template
- `configs/pipeline_deepsets_template.json` - DeepSets pipeline template
- `configs/pipeline_tf_fusion_template.json` - TF-Fusion pipeline template
- `configs/ablation_exp_*.json` - Ablation experiment configs (A_baseline, B_concat_fusion, C_low_physics)
