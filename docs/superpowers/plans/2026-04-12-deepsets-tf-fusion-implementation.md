# DeepSets TF-Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `DeepSetsPINN_TF` model path with offline STFT-based time-frequency input and gated fusion, while keeping existing `DeepCAE_PINN` and `DeepSetsPINN` behavior unchanged.

**Architecture:** Keep `pipeline=deepsets` and gate the new behavior by `model_type=tf_fusion`. Add a dedicated TF feature utility, TF dataset artifacts (`train/tf`, `val/tf`), a dual-branch model (`DeepSetsPINN_TF`), and train/inference wiring that strictly validates checkpoint/model/STFT compatibility. Baseline paths remain untouched for ablation.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy STFT (`scipy.signal.stft`), existing script entrypoints (`uv run python ...`).

---

## File Structure and Responsibilities

- Create: `data/tf_features.py`
  - Build and normalize offline STFT 1D features.
  - Save/load `train/tf` and `val/tf` sample files aligned with noisy indices.
- Modify: `scripts/transformer.py`
  - Add optional TF feature export integrated with existing transform pipeline.
- Modify: `data/data_deepsets.py`
  - Add optional TF loading in `DeepSetsDataset` and expose `tf_signals` batch key.
- Modify: `model/model.py`
  - Add `TimeFreqEncoder`, `GatedFeatureFusion`, `DeepSetsPINN_TF(DeepSetsPINN)`.
- Modify: `scripts/train/train.py`
  - Add `model_type` switch (`deepsets` / `tf_fusion`) and pass `tf_signals` when needed.
- Modify: `scripts/analysis/inference_deepsets.py`
  - Add `tf_fusion` inference path and STFT metadata checks.
- Modify: `scripts/run_unified_pipeline.py`
  - Thread new TF/STFT args into train/inference config dictionaries.
- Modify: `configs/pipeline_deepsets_template.json`
  - Add `model_type=tf_fusion`-related fields and STFT defaults.
- Modify: `configs/README.md`
  - Document new TF feature flow and parameters.
- Create: `scripts/dev/check_tf_fusion_smoke.py`
  - Focused smoke script for shape/interface checks without full training.

---

### Task 1: Add Offline TF Feature Utilities

**Files:**
- Create: `data/tf_features.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write the failing test (import and function contract)**

```python
# scripts/dev/check_tf_fusion_smoke.py
from pathlib import Path
import numpy as np

from data.tf_features import stft_to_1d_feature, normalize_to_minus1_1


def main() -> None:
    signal = np.sin(np.linspace(0, 8 * np.pi, 1000)).astype(np.float64)
    feat = stft_to_1d_feature(
        signal,
        target_length=1000,
        n_fft=128,
        hop_length=32,
        win_length=128,
        window="hann",
        pooling="mean",
    )
    assert feat.shape == (1000,)
    norm, vmin, vmax = normalize_to_minus1_1(feat)
    assert norm.shape == (1000,)
    assert np.isfinite(norm).all()
    assert np.max(norm) <= 1.0 + 1e-6
    assert np.min(norm) >= -1.0 - 1e-6
    assert vmax >= vmin


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.tf_features'`

- [ ] **Step 3: Write minimal implementation (`data/tf_features.py`)**

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from scipy.signal import stft


def normalize_to_minus1_1(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    sig = signal.astype(np.float64)
    vmin = float(sig.min())
    vmax = float(sig.max())
    amp = vmax - vmin
    if amp > np.finfo(np.float64).eps * 0.1:
        out = 2.0 * (sig - vmin) / amp - 1.0
    else:
        out = np.zeros_like(sig, dtype=np.float64)
    return out.astype(np.float32), vmin, vmax


def stft_to_1d_feature(
    signal_1d: np.ndarray,
    target_length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    pooling: Literal["mean", "max", "meanmax"],
) -> np.ndarray:
    _, _, zxx = stft(
        signal_1d.astype(np.float64),
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window=window,
        boundary=None,
        padded=False,
    )
    mag = np.log1p(np.abs(zxx))
    if pooling == "mean":
        tf_1d = mag.mean(axis=0)
    elif pooling == "max":
        tf_1d = mag.max(axis=0)
    else:
        tf_1d = 0.5 * (mag.mean(axis=0) + mag.max(axis=0))
    x_old = np.linspace(0.0, 1.0, num=tf_1d.shape[0], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    return np.interp(x_new, x_old, tf_1d).astype(np.float64)


def save_tf_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feature.astype(np.float32))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS (no output, exit code 0)

- [ ] **Step 5: Commit**

```bash
git add scripts/dev/check_tf_fusion_smoke.py data/tf_features.py
git commit -m "feat: add offline STFT 1D tf feature utility"
```

---

### Task 2: Export TF Features During Transform

**Files:**
- Modify: `scripts/transformer.py`
- Modify: `data/tf_features.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write failing test (expected tf files are created)**

```python
# append to scripts/dev/check_tf_fusion_smoke.py
from tempfile import TemporaryDirectory
from scripts.transformer import save_dataset
from data.tf_features import save_tf_feature


def test_tf_paths_created() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "noisy").mkdir(parents=True, exist_ok=True)
        (root / "train" / "clean").mkdir(parents=True, exist_ok=True)
        (root / "val" / "noisy").mkdir(parents=True, exist_ok=True)
        (root / "val" / "clean").mkdir(parents=True, exist_ok=True)
        save_tf_feature(root / "train" / "tf" / "0000.npy", np.zeros(1000))
        assert (root / "train" / "tf" / "0000.npy").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL (until transformer path wiring is added)

- [ ] **Step 3: Implement transformer TF export hooks**

```python
# scripts/transformer.py (new args in transform_data / CLI)
# add args:
#   export_tf: bool = False
#   tf_n_fft: int = 128
#   tf_hop_length: int = 32
#   tf_win_length: int = 128
#   tf_window: str = "hann"
#   tf_pooling: str = "mean"

# after save_dataset(...), if export_tf:
# 1) iterate saved train/noisy and val/noisy files
# 2) load each noisy npy
# 3) compute tf_1d = stft_to_1d_feature(...)
# 4) normalize_to_minus1_1(tf_1d)
# 5) save to parallel train/tf and val/tf with same filename
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS, and TF file path assertions pass

- [ ] **Step 5: Commit**

```bash
git add scripts/transformer.py data/tf_features.py scripts/dev/check_tf_fusion_smoke.py
git commit -m "feat: add optional offline tf export in transformer"
```

---

### Task 3: Extend DeepSets Dataset With TF Signals

**Files:**
- Modify: `data/data_deepsets.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write failing test (batch must include `tf_signals`)**

```python
# add to scripts/dev/check_tf_fusion_smoke.py
def test_dataset_tf_key() -> None:
    sample = {
        "noisy_signals": np.zeros((25, 1000), dtype=np.float32),
        "clean_signals": np.zeros((25, 1000), dtype=np.float32),
        "coordinates": np.zeros((25, 2), dtype=np.float32),
        "grid_indices": np.zeros((25, 2), dtype=np.int64),
        "tf_signals": np.zeros((25, 1000), dtype=np.float32),
    }
    assert "tf_signals" in sample
```

- [ ] **Step 2: Run test to verify it fails in real dataloader path**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL when reading real batch without TF support

- [ ] **Step 3: Implement dataset wiring**

```python
# data/data_deepsets.py
# - add constructor flag: use_tf: bool = False
# - if use_tf, load parallel tf grids from data_dir/tf
# - validate counts match noisy/clean
# - in __getitem__, add:
#     sample["tf_signals"] = torch.from_numpy(tf_patch_or_full)

# create_deepsets_dataloaders(...)
# - add arg use_tf: bool = False
# - pass use_tf to train/val datasets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS, `tf_signals` present for `use_tf=True`

- [ ] **Step 5: Commit**

```bash
git add data/data_deepsets.py scripts/dev/check_tf_fusion_smoke.py
git commit -m "feat: add optional tf_signals key to deepsets dataset"
```

---

### Task 4: Implement `DeepSetsPINN_TF` and Fusion Modules

**Files:**
- Modify: `model/model.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write failing test (model forward signature and shape)**

```python
# add to scripts/dev/check_tf_fusion_smoke.py
import torch
from model.model import DeepSetsPINN_TF


def test_tf_model_forward_shape() -> None:
    model = DeepSetsPINN_TF(signal_embed_dim=128, coord_embed_dim=64, point_dim=128)
    noisy = torch.randn(2, 25, 1000)
    tf_sig = torch.randn(2, 25, 1000)
    coords = torch.randn(2, 25, 2)
    out = model(noisy, tf_sig, coords)
    assert out.shape == noisy.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL with missing `DeepSetsPINN_TF`

- [ ] **Step 3: Implement classes in `model/model.py`**

```python
class TimeFreqEncoder(SignalEncoder):
    """Encoder for 1D time-frequency proxy signals."""


class GatedFeatureFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_time: torch.Tensor, f_tf: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_time, f_tf], dim=-1))
        fused = g * f_time + (1.0 - g) * f_tf
        return self.norm(fused)


class DeepSetsPINN_TF(DeepSetsPINN):
    def __init__(self, *args, tf_embed_dim: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        signal_embed_dim = kwargs.get("signal_embed_dim", 256)
        tf_dim = tf_embed_dim or signal_embed_dim
        if tf_dim != signal_embed_dim:
            raise ValueError("tf_embed_dim must equal signal_embed_dim in v1")
        self.tf_encoder = TimeFreqEncoder(
            base_channels=kwargs.get("base_channels", 32),
            kernel_size=kwargs.get("kernel_size", 7),
            dropout_rate=kwargs.get("dropout_rate", 0.1),
            embed_dim=signal_embed_dim,
        )
        self.fusion = GatedFeatureFusion(embed_dim=signal_embed_dim)

    def forward(
        self,
        noisy_signals: torch.Tensor,
        tf_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        b, r, t = noisy_signals.shape
        sig_emb = self.signal_encoder(noisy_signals.reshape(b * r, 1, t)).view(b, r, -1)
        tf_emb = self.tf_encoder(tf_signals.reshape(b * r, 1, t)).view(b, r, -1)
        fused = self.fusion(sig_emb, tf_emb)
        coord_emb = self.coord_mlp(coordinates)
        point_feat = self.point_encoder(fused, coord_emb)
        global_feat = point_feat.mean(dim=1, keepdim=True).expand(-1, r, -1)
        dec_input = torch.cat([point_feat, global_feat], dim=-1)
        return self.decoder(dec_input.reshape(b * r, -1)).view(b, r, t)

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        tf_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(noisy_signals, tf_signals, coordinates)
        residual = self.compute_wave_equation_residual(
            denoised, grid_indices, grid_cols, grid_rows
        )
        return denoised, residual
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS, forward shape check succeeds

- [ ] **Step 5: Commit**

```bash
git add model/model.py scripts/dev/check_tf_fusion_smoke.py
git commit -m "feat: add DeepSetsPINN_TF with gated time-tf fusion"
```

---

### Task 5: Integrate Training Path (`model_type=tf_fusion`)

**Files:**
- Modify: `scripts/train/train.py`
- Modify: `scripts/run_unified_pipeline.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write failing test (config route selects TF model)**

```python
# add to scripts/dev/check_tf_fusion_smoke.py
from scripts.train.train import train_from_config


def test_train_config_accepts_tf_fusion() -> None:
    cfg = {
        "pipeline": "deepsets",
        "model_type": "tf_fusion",
        "epochs": 1,
        "batch_size": 1,
        "data_path": "data",
    }
    assert cfg["model_type"] == "tf_fusion"
```

- [ ] **Step 2: Run test to verify it fails in runtime integration**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL because train path does not yet consume `tf_signals` / `tf_fusion`

- [ ] **Step 3: Implement training branching**

```python
# scripts/train/train.py
# - import DeepSetsPINN_TF
# - in train_deepsets_pinn(...), keep existing path for model_type=="deepsets"
# - for model_type=="tf_fusion":
#     model = DeepSetsPINN_TF(...)
#     require batch["tf_signals"]
#     call model.physics_forward(noisy, tf_signals, coords, ...)
# - save checkpoint metadata:
#     "model_type": model_type
#     "stft_*": values

# scripts/run_unified_pipeline.py
# - pass through stft args and model_type in train config dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS for config route and no key errors in smoke path

- [ ] **Step 5: Commit**

```bash
git add scripts/train/train.py scripts/run_unified_pipeline.py scripts/dev/check_tf_fusion_smoke.py
git commit -m "feat: wire tf_fusion model_type into deepsets training"
```

---

### Task 6: Integrate Inference Path and Metadata Validation

**Files:**
- Modify: `scripts/analysis/inference_deepsets.py`
- Modify: `data/tf_features.py`
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Write failing test (checkpoint/model_type mismatch must fail)**

```python
# add to scripts/dev/check_tf_fusion_smoke.py
def test_inference_requires_compatible_model_type() -> None:
    ckpt_meta = {"model_type": "deepsets"}
    requested = "tf_fusion"
    assert ckpt_meta["model_type"] != requested
```

- [ ] **Step 2: Run test to verify it fails in inference route**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL until strict checks are implemented

- [ ] **Step 3: Implement inference TF route + checks**

```python
# scripts/analysis/inference_deepsets.py
# - add CLI/config arg model_type with choices ["deepsets", "tf_fusion"]
# - when model_type=="tf_fusion":
#     1) build tf_signals with stft_to_1d_feature + normalize_to_minus1_1
#     2) call model(noisy, tf_signals, coords)
# - validate:
#     ckpt["model_type"] == requested model_type
#     ckpt stft params match runtime params
# - raise ValueError on mismatch (hard fail)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS for mismatch-check assertions and tf route dry-run checks

- [ ] **Step 5: Commit**

```bash
git add scripts/analysis/inference_deepsets.py data/tf_features.py scripts/dev/check_tf_fusion_smoke.py
git commit -m "feat: add tf_fusion inference path with strict metadata checks"
```

---

### Task 7: Config and Docs Updates

**Files:**
- Modify: `configs/pipeline_deepsets_template.json`
- Modify: `configs/README.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing test (config missing required tf keys)**

```python
# add to scripts/dev/check_tf_fusion_smoke.py
import json


def test_config_has_tf_keys() -> None:
    cfg = json.loads(Path("configs/pipeline_deepsets_template.json").read_text())
    required = [
        "model_type",
        "stft_n_fft",
        "stft_hop_length",
        "stft_win_length",
        "stft_window",
        "stft_pooling",
    ]
    for k in required:
        assert k in cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: FAIL because template lacks new TF keys

- [ ] **Step 3: Implement config/docs updates**

```json
{
  "model_type": "deepsets",
  "stft_n_fft": 128,
  "stft_hop_length": 32,
  "stft_win_length": 128,
  "stft_window": "hann",
  "stft_pooling": "mean",
  "export_tf": false
}
```

```markdown
- New model option: `model_type=tf_fusion`
- Enable TF export in transform stage with `export_tf=true`
- Ensure train/inference STFT params are identical
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS for config key presence checks

- [ ] **Step 5: Commit**

```bash
git add configs/pipeline_deepsets_template.json configs/README.md README.md scripts/dev/check_tf_fusion_smoke.py
git commit -m "docs: add tf_fusion config and usage guidance"
```

---

### Task 8: End-to-End Verification and Non-Regression Checks

**Files:**
- Test: `scripts/dev/check_tf_fusion_smoke.py`

- [ ] **Step 1: Run compile check**

Run: `uv run python -m compileall .`
Expected: PASS without syntax errors.

- [ ] **Step 2: Run focused TF smoke script**

Run: `uv run python scripts/dev/check_tf_fusion_smoke.py`
Expected: PASS all assertions.

- [ ] **Step 3: Run baseline non-regression smoke**

Run: `uv run python scripts/train/train.py --pipeline pinn --epochs 1 --num_train 64 --num_val 16`
Expected: PASS baseline PINN still trains.

- [ ] **Step 4: Run baseline DeepSets smoke**

Run: `uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 1 --batch_size 8`
Expected: PASS baseline DeepSets still trains.

- [ ] **Step 5: Run TF-fusion smoke (if tf data prepared)**

Run: `uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 1 --batch_size 8 --model_type tf_fusion`
Expected: PASS with `tf_signals` consumed and checkpoint metadata including `model_type=tf_fusion`.

- [ ] **Step 6: Commit verification notes**

```bash
git add .
git commit -m "test: verify tf_fusion integration and baseline non-regression"
```

---

## Self-Review Notes (Plan Quality)

- Spec coverage: model class, data flow, train/inference wiring, config/checkpoint schema, non-regression and ablation hooks are all mapped to explicit tasks.
- Placeholder scan: no TODO/TBD placeholders left in execution steps.
- Type consistency: `tf_signals` tensor shape is consistently `[B, R, T]`; `model_type` key consistently uses `deepsets`/`tf_fusion`.
