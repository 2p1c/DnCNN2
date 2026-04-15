# Conventions

## Code Style

- **Language**: Python 3.10+
- **Style**: PEP 8 with type annotations throughout
- **Formatter**: Not explicitly configured (no pre-commit, no pyproject.toml formatter)

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `DeepCAE`, `PINNLoss` |
| Functions/methods | snake_case | `train_one_epoch`, `validate` |
| Constants | UPPER_SNAKE_CASE | `NUM_POINTS`, `WAVE_SPEED` |
| Module-level constants | UPPER_SNAKE_CASE | `SAMPLING_RATE`, `CENTER_FREQ` |
| Variables | snake_case | `noisy_signal`, `clean_signal` |
| Private methods | snake_case with `_` prefix | `_compute_physics_residual` |

## Import Organization

```python
# Standard library
import os
from typing import Optional

# Third-party
import torch
import torch.nn as nn
import numpy as np

# Local
from model.model import DeepCAE, DeepCAE_PINN
from data.data_utils import PINNDataset
```

## Entry Point Pattern

All scripts use the `main()` function pattern:

```python
def main():
    # CLI logic here
    pass

if __name__ == "__main__":
    main()
```

## Error Handling

- Use descriptive error messages
- Catch specific exceptions: `ValueError`, `FileNotFoundError`, `RuntimeError`
- Fail fast with clear messages for invalid inputs

```python
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data directory not found: {data_path}")
```

## Logging

Currently uses print statements with prefixed tags:

```python
print(f"[INFO] Loading data from {data_path}")
print(f"[WARNING] Missing optional dependency")
print(f"[Step 3/5] Computing validation metrics")
```

**Note**: No structured logging library configured.

## File Organization

- **model/**: Model definitions only
- **data/**: Dataset classes and data utilities
- **scripts/**: Entry points and orchestration
- **configs/**: JSON configuration templates
- **results/**: Generated outputs (checkpoints, images, experiments)

## Configuration

- JSON files for pipeline configuration (no YAML, no environment variable loading)
- Key physics constants defined in CLAUDE.md:
  - Sampling Rate: 6.25 MHz
  - Duration: 160 μs
  - Points: 1000
  - Center Frequency: 250 kHz
  - Wave Speed: 5900 m/s (steel)

## Documentation

- Inline comments for non-obvious logic
- No docstrings on all functions (inconsistent)
- CLAUDE.md provides project-level context
- AGENTS.md for agent instructions

## CLI Conventions

- Use argparse for all CLI tools
- Long-form arguments with `--` prefix
- Required arguments positional, optional with `-` prefix

```python
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, default="results/")
parser.add_argument("--checkpoint", type=str, default=None)
```
