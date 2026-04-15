# Concerns

## Critical Issues

### 1. Zero Test Coverage

**Severity**: HIGH

No test files exist in the project. All validation is manual through preview scripts. This creates risk of:
- Undetected regressions in model implementations
- Physics residual bugs without systematic detection
- Data loading edge cases undetected

**Recommendation**: Add pytest framework with unit tests for models, physics residuals, and data loading.

### 2. Security: torch.load with weights_only=False

**Severity**: HIGH

Multiple inference scripts use `torch.load(..., weights_only=False)`:
- `scripts/analysis/inference.py`
- `scripts/analysis/inference_deepsets.py`
- `scripts/analysis/preview_signals.py`

This allows arbitrary code execution from pickled files. If checkpoint files are from untrusted sources, this is a security vulnerability.

**Recommendation**: Use `weights_only=True` or validate checkpoint sources.

### 3. CoordinateMLP Embed Dim Bug

**Severity**: MEDIUM

`model/model.py:288` hardcodes `nn.Linear(2, 64)` instead of using the `embed_dim` parameter:

```python
# BUG: hardcoded 2 → embed_dim
self.fc1 = nn.Linear(2, 64)  # Should be nn.Linear(embed_dim, 64)
```

**Recommendation**: Fix to use `embed_dim` parameter.

## Performance Issues

### 4. DeepSets Batch Processing Inefficiency

**Severity**: MEDIUM

`DeepSetsPINN.denoise_grid()` processes patches one-at-a-time with ~1296 forward passes per signal (36x36 grid):

```python
for i in range(self.grid_size):
    for j in range(self.grid_size):
        output = self.point_net(...)  # One at a time
```

**Recommendation**: Batch processing for GPU efficiency.

### 5. Duplicate Loss Classes

**Severity**: LOW

`scripts/train/train.py:148-175` defines two identical loss classes:

```python
class PINNLoss(nn.Module):     # lines 148-158
class DeepSetsPINNLoss(nn.Module):  # lines 160-175
# Both are identical implementations
```

**Recommendation**: Merge into single `PhysicsLoss` class with pipeline-specific equation selection.

## Code Quality

### 6. Large Files Exceeding Guidelines

**Severity**: LOW

- `scripts/train/train.py`: 1041 lines (guideline: 800 max)
- `data/data_utils.py`: 882 lines (guideline: 800 max)

**Recommendation**: Split into smaller modules by responsibility.

### 7. Debug Print Statements

**Severity**: LOW

Six `[DEBUG]` print statements in `scripts/transformer.py:596-667`:

```python
print(f"[DEBUG] Creating validation split...")
```

**Recommendation**: Replace with proper logging or remove.

### 8. Unused Parameters

**Severity**: LOW

- `scripts/train/train.py:763`: `del model_type, coord_dim` (deleted after assignment)
- `model/model.py:483`: `del grid_indices, grid_cols, grid_rows` (deleted after assignment)

**Recommendation**: Remove unused variable assignments.

## Inconsistencies

### 9. Inconsistent Physics Weight Defaults

| Pipeline | Default physics_weight |
|----------|----------------------|
| PINN | 1e-3 |
| DeepSets | 1e-4 |

**Recommendation**: Document reason for difference or normalize defaults.

### 10. Hardcoded Magic Numbers

Scattered across files:
- `NUM_POINTS=1000` - signal length
- Grid sizes: 21, 41
- `wave_speed=5900` (m/s)
- Sampling rate: 6.25 MHz

**Recommendation**: Centralize in a `constants.py` module.

## Missing Infrastructure

### 11. No Formal Test Framework

See Testing.md - no pytest, no tests directory, no CI test execution.

### 12. No Dependency Locking

`pyproject.toml` may not pin exact versions. `uv sync` creates `uv.lock` but it's not committed.

**Recommendation**: Commit `uv.lock` for reproducible builds.
