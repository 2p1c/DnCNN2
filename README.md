# Ultrasonic Signal Denoising (PINN / DeepSets-PINN)

本项目用于超声 1D 信号去噪（NDT 场景），当前主流程已统一到：

- 训练入口：`scripts/train/train.py`
- 一体化流程入口：`scripts/run_unified_pipeline.py`
- 模型定义入口：`model/model.py`

当前仅保留并使用 3 个核心模型：

- `DeepCAE`
- `DeepCAE_PINN`（继承 `DeepCAE`）
- `DeepSetsPINN`

历史脚本和冗余模型（如 `train_pinn.py`、`train_deepsets_pinn.py`、`SpatialAuxiliaryCAE`、`model_deepsets.py`）已移除。

---

## Environment

```bash
uv sync
```

---

## Standard Output Directories

- Figures: `results/<run_timestamp>/images/`
- Checkpoints: `results/<run_timestamp>/checkpoints/`
- Inference `.mat` outputs: `results/<run_timestamp>/`

其中 `<run_timestamp>` 由 `scripts/run_unified_pipeline.py` 自动创建。

---

## 1) 数据准备（transform）

统一使用 `scripts/transformer.py`，将实验 `.mat` 转换为训练目录结构（`train/` + `val/`）。

### 基本用法

```bash
uv run python scripts/transformer.py \
  --noisy data/noisy.mat \
  --clean data/clean.mat \
  --output data
```

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--noisy` | 必填 | 噪声 `.mat` 文件 |
| `--clean` | 必填 | 干净 `.mat` 文件 |
| `--output` | `data` | 输出目录 |
| `--noisy_cols` / `--noisy_rows` | `21` / `21` | 输入网格尺寸 |
| `--clean_cols` / `--clean_rows` | `41` / `41` | 目标网格尺寸 |
| `--augment_factor` | `5` | 数据增强倍数 |
| `--train_ratio` | `0.8` | 训练集比例 |
| `--interp_method` | `cubic` | 空间插值（`linear` / `cubic`） |
| `--signal_length` | `1000` | 信号长度（超出截断） |
| `--seed` | `42` | 随机种子 |

---

## 2) 统一训练入口（scripts/train/train.py）

训练通过 `--pipeline` 选择分支：

- `pinn`：`DeepCAE_PINN` 路径
- `deepsets`：`DeepSetsPINN` 路径

### 2.1 PINN 训练

```bash
# 合成数据快速测试
uv run python scripts/train/train.py --pipeline pinn --epochs 2

# 文件模式训练
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --epochs 100

# 调节物理约束
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data --physics_weight 0.001
```

常用参数（PINN）：

| 参数 | 默认值 |
|---|---|
| `--mode` | `synthetic` |
| `--data_path` | `data` |
| `--epochs` | `50` |
| `--batch_size` | `32` |
| `--lr` | `0.001` |
| `--dropout` | `0.1` |
| `--patience` / `--min_epochs` | `50` / `30` |
| `--physics_weight` | `1e-3`（默认逻辑） |
| `--wave_speed` | `5900.0` |
| `--center_frequency` | `250000.0` |
| `--damping_ratio` | `0.05` |

典型输出：

- `fig_pinn_pre_train_samples.png`
- `fig_pinn_results.png`
- `fig_pinn_training_curves.png`
- `fig_pinn_acoustic_validation.png`
- `best_pinn_model.pth`

### 2.2 DeepSets-PINN 训练

```bash
# 最小烟测
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 2 --batch_size 8

# 基础训练
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --epochs 100 --patch_size 5

# 调节物理损失权重
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data --physics_weight 1e-4
```

常用参数（DeepSets）：

| 参数 | 默认值 |
|---|---|
| `--grid_cols` / `--grid_rows` | `41` / `41` |
| `--patch_size` / `--stride` | `5` / `1` |
| `--base_channels` | `16` |
| `--coord_dim` | `64` |
| `--signal_embed_dim` | `128` |
| `--coord_embed_dim` | `64` |
| `--point_dim` | `128` |
| `--physics_weight` | `1e-4`（默认逻辑） |
| `--wave_speed` | `5900.0` |
| `--center_frequency` | `250000.0` |

典型输出：

- `fig_deepsets_pinn_pre_train_samples.png`
- `fig_deepsets_pinn_training_curves.png`
- `fig_deepsets_pinn_results.png`
- `fig_deepsets_pinn_acoustic_validation.png`
- `best_deepsets_pinn.pth`

---

## 3) 一体化流程（推荐）

`scripts/run_unified_pipeline.py` 会串联：

1. transform
2. train
3. inference

并支持 JSON 配置（推荐）。

### 3.1 PINN 全流程

```bash
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json
```

### 3.2 DeepSets 全流程

```bash
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_deepsets_template.json
```

### 3.3 跳过 transform/train（复用已有数据和权重）

```bash
uv run python scripts/run_unified_pipeline.py \
  --config configs/pipeline_pinn_template.json \
  --skip_transform \
  --skip_train \
  --inference_input data/noisy.mat \
  --checkpoint results/checkpoints/best_pinn_model.pth
```

DeepSets 对应替换为 `pipeline_deepsets_template.json` 与 `best_deepsets_pinn.pth`。

---

## 4) 独立推理脚本

### 4.1 PINN/CAE 推理

```bash
uv run python scripts/analysis/inference.py \
  --input data/noisy.mat \
  --output results/ \
  --checkpoint results/checkpoints/best_pinn_model.pth
```

### 4.2 DeepSets 推理

```bash
uv run python scripts/analysis/inference_deepsets.py \
  --input data/noisy.mat \
  --output results/ \
  --checkpoint results/checkpoints/best_deepsets_pinn.pth
```

推理输出通常包括：

- PINN：`denoised_<ts>_full.mat` / `denoised_<ts>_original.mat`
- DeepSets：`deepsets_denoised_<ts>.mat` / `deepsets_denoised_<ts>_original.mat`
- 声学验证图：`acoustic_validation*.png`

---

## 5) 调参建议

### PINN

| 现象 | 建议 |
|---|---|
| 物理约束过强（PSNR 低） | 降低 `physics_weight`（如 `1e-4`） |
| 物理约束太弱（physics_loss 不降） | 适度提高 `physics_weight`（如 `1e-3 ~ 1e-2`） |
| 输出过平滑 | 降低 `physics_weight`，或降低 `dropout` |
| 训练不稳定 | 降低 `lr`（如 `5e-4`） |

### DeepSets

| 现象 | 建议 |
|---|---|
| 收敛慢/不稳定 | 降低 `lr`，增大 `patience` |
| 去噪细节变差 | 保持 `patch_size=5` 起步，再小步调参 |
| 结果偏向过强平滑 | 先降低 `physics_weight` |

---

## 6) 核心信号设置

| Parameter | Value |
|---|---|
| Sampling Rate | 6.25 MHz |
| Duration | 160 us |
| Data Points | 1000 |
| Center Frequency | 250 kHz |

---

## 7) Project Structure (Simplified)

```text
DnCNN2/
├── model/
│   ├── model.py
│   └── model_networks_overview.md
├── data/
│   ├── data_utils.py
│   └── data_deepsets.py
├── scripts/
│   ├── transformer.py
│   ├── run_unified_pipeline.py
│   ├── train/
│   │   └── train.py
│   └── analysis/
│       ├── inference.py
│       ├── inference_deepsets.py
│       ├── acoustic_validation.py
│       └── preview_signals.py
├── configs/
│   ├── pipeline_pinn_template.json
│   └── pipeline_deepsets_template.json
└── README.md
```

---

## License

MIT
