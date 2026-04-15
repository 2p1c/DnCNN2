# Testing

## Test Infrastructure Status

**No formal test framework detected.** The project has zero test files.

## Validation Approach

Validation is performed through ad-hoc scripts rather than automated tests:

### Acoustic Validation

```bash
uv run python scripts/analysis/acoustic_validation.py
```

Validates acoustic physics properties of denoised signals.

### Signal Preview

```bash
uv run python scripts/analysis/preview_signals.py --detailed --no_show
```

Visual inspection of signal quality.

### RRMSE Metric

Both inference scripts (`inference.py`, `inference_deepsets.py`) compute Relative Root Mean Square Error (RRMSE) as the primary quantitative metric:

```python
rrmse = torch.sqrt(torch.mean((denoised - clean) ** 2)) / torch.sqrt(torch.mean(clean ** 2))
```

## CI Pipeline

`.github/workflows/ci.yml` only deploys documentation via MkDocs:

```yaml
- uses: actions/checkout@v4
- uses: ./.github/actions/setup
- run: mkdocs gh-deploy --force
```

No test execution in CI.

## Coverage Gaps

| Area | Status |
|------|--------|
| Model forward pass | No tests |
| Physics residual computation | No tests |
| Data loading | No tests |
| Data augmentation | Manual only |
| PINNLoss | No tests |
| DeepSetsPINNLoss | No tests |
| CoordinateMLP | No tests |
| GridEncoder | No tests |
| Training loop | Manual only |
| Inference | Manual only |

## Recommendations

1. Add pytest with `pytest-cov` for unit testing
2. Test physics residual computations against known analytical solutions
3. Test model output shape consistency
4. Test data loading and train/val split correctness
5. Add integration tests for full pipeline execution
6. Configure CI to run tests on push
