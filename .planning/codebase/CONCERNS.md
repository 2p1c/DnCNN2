# Codebase Concerns

**Analysis Date:** 2026-04-24

## Tech Debt

**File Size Violations:**
- `scripts/train/train.py`: 1246 lines (guideline: 800 max)
- `scripts/train/train_gui.py`: 1229 lines (guideline: 800 max) - NEW FILE
- `data/data_utils.py`: 881 lines (guideline: 800 max)

The new `train_gui.py` exceeds the file size guideline immediately upon creation. Split into smaller modules by responsibility.

**CoordinateMLP Embed Dim Bug:**
- File: `model/model.py:292`
- Issue: Hardcoded `nn.Linear(2, 64)` ignores the `embed_dim` parameter
- Impact: Coordinate embedding dimension cannot be configured
- Fix approach: Change to `nn.Linear(2, embed_dim)` and verify downstream usage

**Duplicate Loss Classes:**
- Files: `scripts/train/train.py:148-175`
- Issue: `PINNLoss` and `DeepSetsPINNLoss` are identical implementations
- Fix approach: Merge into single `PhysicsLoss` class with pipeline-specific equation selection

**Debug Print Statements:**
- File: `scripts/transformer.py:606-677`
- Issue: Six `[DEBUG]` print statements remain after development
- Fix approach: Remove or replace with proper logging

**Unused Variable Deletions:**
- `scripts/train/train.py:836`: `del coord_dim` (after assignment, never used)
- `model/model.py:531`: `del grid_indices, grid_cols, grid_rows` (after assignment, never used)
- Fix approach: Remove these no-op deletions

## Known Bugs

**Weights_Only Security Risk:**
- Files affected:
  - `scripts/analysis/inference.py:104`
  - `scripts/analysis/inference_deepsets.py:81`
  - `scripts/train/train.py:673,772,900,1032`
- Issue: `torch.load(..., weights_only=False)` allows arbitrary code execution from pickled checkpoints
- Current mitigation: Comment notes this is "needed for checkpoints saved with additional metadata"
- Fix approach: Create a validated checkpoint loader that verifies checkpoint integrity before loading

## Performance Bottlenecks

**MPS Fallback When CUDA Unavailable:**
- File: `scripts/train/train.py:596-605`
- Issue: Device priority is CUDA > MPS > CPU, but MPS performance on Apple Silicon is significantly slower than CUDA for these workloads
- Current capacity: Limited by Metal Performance Shaders overhead
- Scaling path: Consider batching for MPS or requiring CUDA for large-scale training

**Signal Length Enforcement:**
- File: `scripts/transformer.py:610-619`
- Issue: All signals forcibly truncated/extended to 1000 points
- Impact: Any non-standard signal length requires transformer processing before use

## Security Considerations

**Checkpoint Loading:**
- Risk: `weights_only=False` on untrusted checkpoint files
- Files: `scripts/analysis/inference*.py`, `scripts/train/train.py`
- Current mitigation: None - checkpoints assumed trusted
- Recommendations:
  1. Add checkpoint hash verification before loading
  2. Document that checkpoints must come from trusted sources
  3. Consider signing checkpoints with a project key

## Branch-Specific Concerns (feat/timefredomine-channel)

**Recent Changes Impact:**
- `scripts/train/train_gui.py` (1229 lines) - New GUI training module added
- `scripts/train/train.py` grew by ~275 lines with tf_fusion support
- `model/model.py` grew by ~152 lines with `DeepSetsPINN_TF` class
- `configs/pipeline_tf_fusion.json` - Config with hardcoded absolute paths:
  - `/Volumes/ESD-ISO/数据/260324/...` (macOS volume path)
  - `/Volumes/ESD-ISO/数据/260407/...` (macOS volume path)
- Issue: These paths will break on any system other than the developer's machine
- Fix approach: Use relative paths or environment variables for data locations

**Hardcoded Data Paths:**
- `configs/pipeline_tf_fusion.json` contains absolute paths to external volumes
- These should be replaced with environment variables or relative paths
- Configs should be gitignore'd or template-only

## Test Coverage Gaps

**No Test Framework:**
- Issue: No pytest configuration, no test files, no tests directory
- What's not tested:
  - Model forward pass correctness
  - Physics residual computation
  - Data loading edge cases (empty grids, mismatched sizes)
  - Loss function calculations
- Risk: Undetected regressions in model implementations and physics residuals
- Priority: HIGH

**No Validation Framework:**
- Acoustic validation runs manually via `scripts/analysis/acoustic_validation.py`
- No automated quality gates
- No threshold-based pass/fail criteria

## Missing Critical Features

**No Reproducible Training:**
- `uv.lock` not committed
- Results depend on exact dependency versions
- Fix: Commit `uv.lock` for reproducible builds

**No CI/CD Pipeline:**
- No automated testing on commit
- No build verification
- No deployment pipeline

## Fragile Areas

**Transformer Grid Assertions:**
- File: `scripts/transformer.py:628-638`
- Issue: Hard assertions on grid sizes that crash on malformed input
- Safe modification: Add descriptive error messages, validate before reshape
- Test coverage: No test for malformed input handling

**DeepSetsPINN_TF Forward Pass:**
- File: `model/model.py:634-659`
- Why fragile: Three `_check_finite` calls inline, complex shape handling for tf_signals
- Safe modification: Add validation before fusion, test with NaN/Inf inputs
- Test coverage: No test for numerical stability

---

*Concerns audit: 2026-04-24*
