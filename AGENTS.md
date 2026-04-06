# AGENTS.md for DnCNN2

High-signal guidance for coding agents in this repo.

## Scope and Stack

- Python project `ultrasonic-cae` (`requires-python >=3.10`), managed with `uv`.
- Domain: ultrasonic 1D denoising with 2 active pipelines: `pinn` and `deepsets`.
- Core entrypoints:
  - Unified flow: `scripts/run_unified_pipeline.py`
  - Training only: `scripts/train/train.py`
  - Inference only: `scripts/analysis/inference.py`, `scripts/analysis/inference_deepsets.py`
  - Data transform: `scripts/transformer.py`
  - Models: `model/model.py`

## Instruction Sources To Respect

- Keep and follow: `CLAUDE.md`, `.github/copilot-instructions.md`, `copilot-instructions.md`.
- These emphasize understanding-first, no guessing on ambiguous intent, and maintainable incremental edits.
- Ignore generic web-dev advice in those files when it conflicts with this Python/ML repo.

## Setup and Canonical Commands

```bash
uv sync
uv sync --group dev
```

Run scripts from repo root with:

```bash
uv run python <script>.py [args]
```

Recommended verification (no dedicated linter/pytest config exists yet):

```bash
uv run python -m compileall .
uv run python scripts/analysis/acoustic_validation.py
uv run python scripts/analysis/preview_signals.py --detailed --no_show
uv build
```

## Focused Checks ("Single Test" Equivalents)

```bash
# Fast synthetic sanity training
uv run python scripts/train/train.py --pipeline pinn --epochs 1 --num_train 64 --num_val 16

# File-mode training sanity
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --epochs 1

# Single-path inference sanity
uv run python scripts/analysis/inference.py --input data/noisy.mat --output results/

# Pipeline inference using existing data/checkpoint
uv run python scripts/run_unified_pipeline.py \
  --pipeline pinn --inference_input data/noisy.mat --skip_transform --skip_train \
  --checkpoint results/checkpoints/best_pinn_model.pth
```

## Workflow and CLI Gotchas

- `run_unified_pipeline.py` always requires `--inference_input` (or in JSON config), even when training.
- When `--skip_transform` is set, `--data_dir` must already contain:
  `train/noisy`, `train/clean`, `val/noisy`, `val/clean`.
- `physics_weight` default is pipeline-specific when omitted:
  - `pinn`: `1e-3`
  - `deepsets`: `1e-4`
- `scripts/analysis/inference.py` enforces `--signal_length == 1000` and raises on mismatch.
- Device selection is explicit in train/inference scripts: prefer `cuda`, then `mps`, then `cpu`; keep this behavior.

## Data, Outputs, and Artifacts

- Unified pipeline creates timestamped run folders under `results/<timestamp>/` with:
  - `checkpoints/`
  - `images/`
  - optional `experiments/` markdown records (`--log_experiment`)
- By default `.gitignore` excludes large/generated artifacts (`data/`, `results/`, checkpoints like `*.pth`, and `uv.lock`).
  Do not assume these paths are tracked in git.

## Coding Conventions To Preserve

- Keep changes minimal and local; avoid unrelated refactors.
- Preserve script-style import bootstrapping (`sys.path.append(...)`) unless you are doing a deliberate import-structure migration.
- Validate paths/shape assumptions early and fail with explicit errors.
- Keep canonical signal constants aligned with model defaults unless explicitly changing physics assumptions:
  - sampling rate `6.25e6`
  - duration `160e-6`
  - points `1000`
  - center frequency `250e3`
