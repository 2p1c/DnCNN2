# AGENTS.md for DnCNN2

Guide for coding agents operating in this repository.
Goal: reliable execution, consistent style, and minimal surprises.

## Project Snapshot

- Package: `ultrasonic-cae`
- Python: `>=3.10`
- Dependency manager and task runner: `uv`
- Build backend: `hatchling` (`pyproject.toml`)
- Domain: ultrasonic 1D signal denoising (`CAE`, `PINN`, `DeepSets + PINN`)

## Instruction Hierarchy

Use this precedence when instructions conflict:

1. Direct user request
2. Repository policy files (`CLAUDE.md`, Copilot files, this `AGENTS.md`)
3. Tool or platform defaults

If requirements are unclear, clarify first instead of guessing.

## Policy Files Discovered

Primary policy/config files present:

- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `copilot-instructions.md`

Cursor rules status:

- `.cursorrules`: not present
- `.cursor/rules/`: not present

Copilot policy highlights to preserve:

- Understanding-first workflow; do not guess ambiguous intent.
- Ask focused clarifying questions when information is insufficient.
- Prioritize readability, modularity, and robust error handling.
- Explain reasoning (why), not only mechanics (what).
- Prefer gradual, maintainable changes over large rewrites.

Note: Copilot files include broad web-development guidance; for this repo,
prioritize Python/ML workflow and repository-specific conventions.

## Environment Setup

Run from repository root (`/Users/zyt/ANW/DnCNN2`):

```bash
uv sync
```

Optional development dependencies:

```bash
uv sync --group dev
```

General execution pattern:

```bash
uv run python <script>.py [args]
```

## Build, Lint, and Test Commands

Current state: no official `pytest` suite and no dedicated linter target.
Use the following commands as canonical checks.

### Build

```bash
uv build
uv run python -m compileall .
```

### Lint / Static Validation

```bash
# Lint config is not defined yet; use syntax compilation as baseline.
uv run python -m compileall .
```

### Functional Validation

```bash
uv run python scripts/analysis/acoustic_validation.py
uv run python scripts/analysis/preview_signals.py --detailed --no_show
```

## Running a Single Test (Important)

Because there is no unit-test framework yet, a "single test" means running one
focused command for one behavior.

Recommended focused checks:

```bash
# Fast sanity training pass
uv run python scripts/train/train.py --epochs 1 --num_train 64 --num_val 16

# Single-file inference path
uv run python scripts/analysis/inference.py --input noisy.mat --output results/

# Pipeline-only inference path (skip transform/train)
uv run python scripts/run_unified_pipeline.py \
  --pipeline pinn --inference_input data/noisy.mat --skip_transform --skip_train
```

If `pytest` is introduced later, use this pattern:

```bash
uv run pytest path/to/test_file.py::test_name -q
```

## Frequently Used Commands

```bash
# Training
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data

# Analysis / inference
uv run python scripts/analysis/inference.py --input noisy.mat --output results/
uv run python scripts/analysis/inference_deepsets.py --input noisy.mat --output results/

# Unified pipeline
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json
```

## Code Style Guidelines

### Imports and Module Structure

- Order imports: standard library, third-party, local modules.
- Separate import groups with one blank line.
- Keep script entrypoints explicit; avoid hidden side effects at import time.
- Preserve existing `sys.path` bootstrapping in script-style files unless doing a full import cleanup.

### Formatting and Naming

- Follow PEP 8 with 4-space indentation.
- Keep lines around 100 characters where practical.
- Use descriptive names and small, composable helper functions.
- Naming conventions: `snake_case` (functions/variables), `PascalCase` (classes),
  `UPPER_CASE` (constants).

### Types and Documentation

- Add type hints for public functions and non-trivial internal helpers.
- Prefer explicit container types (`dict[str, float]`, `list[np.ndarray]`, etc.) when useful.
- Document tensor/array shape expectations in docstrings.
- For numerical code, include units/ranges when they are domain-relevant.

### Error Handling and Validation

- Validate file paths, dimensions, and value ranges early.
- Raise specific exceptions with actionable messages.
- Avoid bare `except:`; catch expected exception types.
- Fail loudly on invalid model/checkpoint/data assumptions.
- Keep logs concise and structured (`[INFO]`, `[WARNING]`, `[ERROR]`).

### Numerical and ML-Specific Guardrails

- Preserve canonical signal settings unless change is intentional and documented:
  - sampling rate: `6.25e6`
  - duration: `160e-6`
  - points: `1000`
  - center frequency: `250e3`
- Keep device selection explicit (`cuda`, `mps`, fallback `cpu`).
- Use `torch.no_grad()` for inference and validation code paths.
- Guard against unstable operations (near-zero divide/log, unchecked interpolation shapes).

## Change Management for Agents

- Prefer minimal, targeted edits aligned with existing patterns.
- Do not refactor unrelated modules while implementing focused changes.
- When touching scripts, keep CLI arguments backward compatible when possible.
- Report: files changed, commands run, and key outcomes.
- Use conventional commit prefixes when committing:
  `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`.
