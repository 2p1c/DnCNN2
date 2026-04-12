"""
DeepSets Dataset for Ultrasonic Signal Denoising

Organizes the 41×41 scanning grid signals as DeepSets input,
attaching 2D spatial coordinates (x, y) to each receiver point
and extracting local patches for memory-efficient training.

Data Layout Assumption:
    data/{train,val}/{noisy,clean}/ contains .npy files named
    0000.npy, 0001.npy, ... in row-scan order (row-major, i.e.
    file index = col * n_rows + row).

Usage:
    # Patch mode (5×5 neighbourhood, default)
    dataset = DeepSetsDataset('data/train', grid_cols=41, grid_rows=41)

    # Full-grid mode
    dataset = DeepSetsDataset('data/val', grid_cols=41, grid_rows=41,
                              patch_size=None)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict


# ============================================================
# Physical Constants (must stay in sync with model.py)
# ============================================================
GRID_SPACING: float = 1e-3  # 1 mm physical distance between grid points


class DeepSetsDataset(Dataset):
    """
    Dataset that loads a scanning grid of ultrasonic signals and
    returns local patches (or the full grid) with 2D coordinates.

    Each sample is a *set* of R receiver observations:
        {
            noisy_signals:  (R, T)   float32, normalised to [-1, 1]
            clean_signals:  (R, T)   float32, normalised to [-1, 1]
            coordinates:    (R, 2)   float32, normalised to [0, 1]
            grid_indices:   (R, 2)   int64,   (col, row) in [0, n_cols) × [0, n_rows)
        }

    Args:
        data_dir:     Root of one split, e.g. 'data/train'.
                      Must contain noisy/ and clean/ subdirectories.
        grid_cols:    Number of columns in the scanning grid (x-direction).
        grid_rows:    Number of rows in the scanning grid (y-direction).
        patch_size:   Side length of the square patch to extract.
                      Set to None to return the entire grid as one sample.
                      Default: 5 (→ 25 elements per set).
        stride:       Stride for patch centre extraction. Default: 1
                      (every interior point is a centre). Increase to
                      reduce dataset size.
        signal_length: Expected signal length per file. Default: 1000.
        dx:           Physical grid spacing in x-direction (metres).
                      Default: 1e-3 (1 mm).
        dy:           Physical grid spacing in y-direction (metres).
                      Default: 1e-3 (1 mm).
        augment:      Enable patch-level data augmentation (random
                      time-reversal and jitter). Default: False.
    """

    def __init__(
        self,
        data_dir: str,
        grid_cols: int = 41,
        grid_rows: int = 41,
        patch_size: Optional[int] = 5,
        stride: int = 1,
        signal_length: int = 1000,
        dx: float = GRID_SPACING,
        dy: float = GRID_SPACING,
        augment: bool = False,
        use_tf: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.patch_size = patch_size
        self.stride = stride
        self.signal_length = signal_length
        self.dx = dx
        self.dy = dy
        self.augment = augment
        self.use_tf = use_tf
        if self.use_tf and self.augment:
            print(
                "[DeepSetsDataset] use_tf=True: disabling patch augmentation to keep noisy/tf pairs aligned"
            )
            self.augment = False

        self.n_grid = grid_cols * grid_rows  # 1681 for 41×41

        # ----------------------------------------------------------
        # Load ALL available grid signals.  If the directory has
        # more files than n_grid (e.g. 6724 = 4 × 1681 augmented),
        # treat every consecutive n_grid block as a separate grid
        # copy, multiplying the training set size.
        # ----------------------------------------------------------
        self.grids_noisy, self.grids_clean, self.grids_tf = self._load_all_grids()

        # ----------------------------------------------------------
        # Build normalised coordinate array  (n_grid, 2)
        # Coordinates are normalised to [0, 1] for the model input.
        # col index → x,  row index → y.
        # ----------------------------------------------------------
        cols = np.arange(grid_cols)
        rows = np.arange(grid_rows)
        cc, rr = np.meshgrid(cols, rows, indexing="ij")  # (cols, rows)
        # Flatten in column-major order matching file layout
        self.coords = np.stack(
            [
                cc.flatten() / max(grid_cols - 1, 1),
                rr.flatten() / max(grid_rows - 1, 1),
            ],
            axis=-1,
        ).astype(np.float32)  # (n_grid, 2)

        # Integer grid indices (col, row) — for spatial diff lookup
        self.grid_idx = np.stack([cc.flatten(), rr.flatten()], axis=-1).astype(
            np.int64
        )  # (n_grid, 2)

        # ----------------------------------------------------------
        # Build sample index list
        # ----------------------------------------------------------
        if self.patch_size is not None:
            self.centres = self._build_patch_centres()
        else:
            # Full-grid mode: single sample
            self.centres = None

        # Total samples = patches_per_grid × n_grids
        self.n_grids = len(self.grids_noisy)

    # ==============================================================
    # Data Loading
    # ==============================================================

    def _load_all_grids(self):
        """
        Load ALL .npy files from noisy/ and clean/ subdirectories.

        If the directory contains more than n_grid files, each
        consecutive n_grid block is treated as a separate grid.
        This supports augmented training directories.

        Returns:
            (list_of_noisy, list_of_clean, list_of_tf_or_none)
            Each list element has shape (n_grid, signal_length).
        """
        noisy_dir = self.data_dir / "noisy"
        clean_dir = self.data_dir / "clean"
        tf_dir = self.data_dir / "tf"

        if not noisy_dir.exists() or not clean_dir.exists():
            raise FileNotFoundError(f"Expected noisy/ and clean/ under {self.data_dir}")
        if self.use_tf and not tf_dir.exists():
            raise FileNotFoundError(
                f"Expected tf/ under {self.data_dir} when use_tf=True"
            )

        # Count available files
        n_files = len(list(noisy_dir.glob("*.npy")))
        n_clean_files = len(list(clean_dir.glob("*.npy")))
        if n_files != n_clean_files:
            raise ValueError(
                f"noisy/clean file count mismatch under {self.data_dir}: "
                f"noisy={n_files}, clean={n_clean_files}"
            )
        if self.use_tf:
            n_tf_files = len(list(tf_dir.glob("*.npy")))
            if n_files != n_tf_files:
                raise ValueError(
                    f"noisy/tf file count mismatch under {self.data_dir}: "
                    f"noisy={n_files}, tf={n_tf_files}"
                )

        n_grids = max(1, n_files // self.n_grid)
        n_to_load = n_grids * self.n_grid
        if n_to_load != n_files:
            raise ValueError(
                f"File count {n_files} in {noisy_dir} is not divisible by grid size {self.n_grid}"
            )

        noisy_all = []
        clean_all = []
        tf_all = []
        for i in range(n_to_load):
            fname = f"{i:04d}.npy"
            noisy_all.append(np.load(noisy_dir / fname))
            clean_all.append(np.load(clean_dir / fname))
            if self.use_tf:
                tf_all.append(np.load(tf_dir / fname))

        noisy_all = np.stack(noisy_all, axis=0).astype(np.float32)
        clean_all = np.stack(clean_all, axis=0).astype(np.float32)
        if self.use_tf:
            tf_all = np.stack(tf_all, axis=0).astype(np.float32)
            if tf_all.shape != noisy_all.shape:
                raise ValueError(
                    "tf shape mismatch with noisy data: "
                    f"{tf_all.shape} vs {noisy_all.shape}"
                )

        # Split into grid-sized blocks
        grids_noisy = [
            noisy_all[i * self.n_grid : (i + 1) * self.n_grid] for i in range(n_grids)
        ]
        grids_clean = [
            clean_all[i * self.n_grid : (i + 1) * self.n_grid] for i in range(n_grids)
        ]
        grids_tf = None
        if self.use_tf:
            grids_tf = [
                tf_all[i * self.n_grid : (i + 1) * self.n_grid] for i in range(n_grids)
            ]

        print(
            f"[DeepSetsDataset] Loaded {n_to_load} signals "
            f"({n_grids} grid(s)) from {self.data_dir}"
        )
        print(
            f"  Grid: {self.grid_cols}×{self.grid_rows}, "
            f"Signal length: {self.signal_length}"
        )
        return grids_noisy, grids_clean, grids_tf

    # ==============================================================
    # Patch Centre Extraction
    # ==============================================================

    def _build_patch_centres(self):
        """
        Build a list of valid (col, row) centre positions for patches.

        A patch of size P×P around centre (ci, cj) covers:
            col ∈ [ci - P//2, ci + P//2]
            row ∈ [cj - P//2, cj + P//2]

        Only centres whose full patch lies within the grid are kept.

        Returns:
            np.ndarray of shape (N, 2) with (col, row) centres.
        """
        half = self.patch_size // 2
        centres = []
        for ci in range(half, self.grid_cols - half, self.stride):
            for cj in range(half, self.grid_rows - half, self.stride):
                centres.append((ci, cj))
        centres = np.array(centres, dtype=np.int64)
        print(
            f"  Patch size: {self.patch_size}×{self.patch_size}, "
            f"Stride: {self.stride}, "
            f"Total patches: {len(centres)}"
        )
        return centres

    # ==============================================================
    # Index Helpers
    # ==============================================================

    def _grid_flat_index(self, col: int, row: int) -> int:
        """Convert (col, row) to flat file index (column-major)."""
        return col * self.grid_rows + row

    def _extract_patch_indices(self, centre_col: int, centre_row: int) -> np.ndarray:
        """
        Get flat indices of all grid points in a P×P patch centred at
        (centre_col, centre_row).

        Returns:
            1D array of flat indices, length P*P.
        """
        half = self.patch_size // 2
        indices = []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                indices.append(self._grid_flat_index(centre_col + dc, centre_row + dr))
        return np.array(indices, dtype=np.int64)

    # ==============================================================
    # Dataset Interface
    # ==============================================================

    def __len__(self) -> int:
        if self.centres is not None:
            return len(self.centres) * self.n_grids
        return self.n_grids  # full-grid mode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.centres is not None:
            patches_per_grid = len(self.centres)
            grid_id = idx // patches_per_grid
            patch_id = idx % patches_per_grid
            ci, cj = self.centres[patch_id]
            flat_idx = self._extract_patch_indices(ci, cj)
        else:
            grid_id = idx
            flat_idx = np.arange(self.n_grid)

        noisy = self.grids_noisy[grid_id][flat_idx].copy()
        clean = self.grids_clean[grid_id][flat_idx].copy()
        tf_signals = None
        if self.use_tf and self.grids_tf is not None:
            tf_signals = self.grids_tf[grid_id][flat_idx].copy()

        # Patch-level augmentation (training only)
        if self.augment:
            # Random time-reversal (50% chance)
            if np.random.random() < 0.5:
                noisy = noisy[:, ::-1].copy()
                clean = clean[:, ::-1].copy()
            # Small Gaussian jitter on noisy (σ=0.01)
            noisy = noisy + np.random.normal(0, 0.01, noisy.shape).astype(np.float32)

        sample = {
            "noisy_signals": torch.from_numpy(noisy),  # (R, T)
            "clean_signals": torch.from_numpy(clean),  # (R, T)
            "coordinates": torch.from_numpy(self.coords[flat_idx]),  # (R, 2)
            "grid_indices": torch.from_numpy(self.grid_idx[flat_idx]),  # (R, 2)
        }
        if tf_signals is not None:
            sample["tf_signals"] = torch.from_numpy(tf_signals)
        return sample


# ============================================================
# DataLoader Factory
# ============================================================


def create_deepsets_dataloaders(
    data_root: str = "data",
    grid_cols: int = 41,
    grid_rows: int = 41,
    patch_size: int = 5,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 0,
    dx: float = GRID_SPACING,
    dy: float = GRID_SPACING,
    augment: bool = True,
    use_tf: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders for DeepSets.

    Args:
        data_root:   Root data directory containing train/ and val/.
        grid_cols:   Scanning grid columns.
        grid_rows:   Scanning grid rows.
        patch_size:  Square patch side length.
        stride:      Patch extraction stride.
        batch_size:  Batch size.
        num_workers: DataLoader worker count.
        dx:          Physical x spacing (m).
        dy:          Physical y spacing (m).
        augment:     Enable augmentation for training set.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = DeepSetsDataset(
        data_dir=f"{data_root}/train",
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        patch_size=patch_size,
        stride=stride,
        dx=dx,
        dy=dy,
        augment=augment,
        use_tf=use_tf,
    )
    val_ds = DeepSetsDataset(
        data_dir=f"{data_root}/val",
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        patch_size=patch_size,
        stride=stride,
        dx=dx,
        dy=dy,
        augment=False,  # Never augment validation
        use_tf=use_tf,
    )

    # Detect if MPS — pin_memory not supported on Apple Silicon
    import torch as _torch

    _use_pin = _torch.cuda.is_available()  # only pin for CUDA

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_use_pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_use_pin,
    )

    print(
        f"\n[DataLoader] Train: {len(train_ds)} patches, "
        f"{len(train_loader)} batches (bs={batch_size})"
    )
    print(
        f"[DataLoader] Val:   {len(val_ds)} patches, "
        f"{len(val_loader)} batches (bs={batch_size})"
    )
    return train_loader, val_loader


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSets Dataset — Self-Test")
    print("=" * 60)

    # Try loading validation set (has exactly 1681 files)
    try:
        ds = DeepSetsDataset("data/val", grid_cols=41, grid_rows=41, patch_size=5)
        sample = ds[0]
        print(f"\nSample 0:")
        for k, v in sample.items():
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

        # Quick batch test
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        print(f"\nBatch:")
        for k, v in batch.items():
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

        print("\n✓ Self-test passed!")
    except FileNotFoundError as e:
        print(f"\n⚠ Data not found: {e}")
        print("  Generate data first with: uv run python transformer.py ...")
