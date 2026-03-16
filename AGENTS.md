# AGENTS.md for DnCNN2

Guide for agentic coding assistants operating in this repository.
Focus: reliable commands, repo conventions, and instruction hierarchy.

## Project Snapshot

- Package: `ultrasonic-cae`
- Python: `>=3.10`
- Dependency manager/runner: `uv`
- Build backend: `hatchling` (via `pyproject.toml`)
- Domain: ultrasonic signal denoising (CAE, PINN, DeepSets+PINN)

## Instruction Sources (Policy Files)

Follow these files when making changes:

- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `copilot-instructions.md`

Cursor rules check:

- No `.cursorrules` file exists.
- No `.cursor/rules/` directory exists.

Practical behavior from these policies:

- Understand first; do not guess unclear requirements.
- Prefer readability and modular design.
- Add explicit error handling around risky operations.
- Explain rationale, not only mechanics.

## Environment Setup

From repository root:

```bash
uv sync
```

Optional dev group:

```bash
uv sync --group dev
```

Run scripts with:

```bash
uv run python <script>.py [args]
```

## Build / Lint / Test Commands

Current reality: no configured `pytest` suite and no dedicated linter config.
Use these as canonical validation commands.

### Build

```bash
uv build
uv run python -m compileall .
```

### Lint

```bash
# No official lint target exists yet.
uv run python -m compileall .
```

### Test / Validation

```bash
uv run python scripts/analysis/acoustic_validation.py
uv run python scripts/analysis/preview_signals.py --detailed --no_show
```

## Running a Single Test (Important)

There is no unit-test framework today.
So "single test" means one focused command validating one behavior.

Examples:

```bash
uv run python scripts/train/train.py --epochs 1 --num_train 64 --num_val 16
uv run python scripts/analysis/inference.py --input noisy.mat --output results/
```

If pytest is introduced later, use:

```bash
uv run pytest path/to/test_file.py::test_name -q
```

## Common Commands

```bash
# Training
uv run python scripts/train/train.py
uv run python scripts/train/train_pinn.py
uv run python scripts/train/train_deepsets_pinn.py --data_path data

# Inference
uv run python scripts/analysis/inference.py --input noisy.mat --output results/
uv run python scripts/analysis/inference_deepsets.py --input noisy.mat --output results/

# Unified pipeline (skip transform/train)
uv run python scripts/run_unified_pipeline.py --pipeline pinn --inference_input data/noisy.mat --skip_transform --skip_train
```

## Code Style Guidelines

### Imports and Structure

- Order imports: standard library -> third-party -> local imports.
- Separate import groups with one blank line.
- Keep `sys.path.append(...)` bootstrapping in script entrypoints unless fully refactoring imports.

### Formatting and Naming

- Follow PEP 8, 4-space indentation, and lines around <= 100 chars.
- Prefer small helpers over long, monolithic functions.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.

### Types and Documentation

- Add type hints on public APIs and key internals.
- Prefer explicit `Optional`, `Tuple`, `Dict`, `Literal` when helpful.
- Document tensor/array shapes and assumptions in docstrings.

### Error Handling and Logging

- Validate paths, shapes, and numeric ranges early.
- Raise specific exceptions with actionable messages.
- Avoid bare `except:` and silent failure paths.
- Keep script output concise with `[INFO]` / `[WARNING]` style messages.

### Numerical/Model Rules

- Preserve canonical constants unless intentionally changed and documented:
  - sampling rate `6.25e6`
  - duration `160e-6`
  - points `1000`
  - center frequency `250e3`
- Keep device selection explicit (`cuda`, `mps`, fallback `cpu`).
- Use `torch.no_grad()` for inference/validation.
- Guard unstable operations (near-zero divide/log) and shape-check interpolation/reshape flows.

## Agent Workflow and Commits

- Restate requested changes, ask one focused clarification when ambiguity matters, then make minimal edits.
- Run at least one relevant validation command and report changed files plus commands executed.
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`.
