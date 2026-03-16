# Spatial-Auxiliary DeepCAE 超声信号去噪训练指南

## 架构概述

**SpatialAuxiliaryCAE** 是一个以 1D 卷积编解码器为主体，以空间结构为辅助的神经网络：

- **主线特征提取 (Core)**：使用 5 层 1D Conv Encoder 和 5 层 1D ConvTranspose Decoder 独立处理输入 Patch 中每一个点的 1000 维超声波形。
- **空间辅助微调 (Auxiliary)**：在 Encoder 最深的潜空间（Latent Space），汇集由 MaxPool 提取的全局邻域上下文以及本地坐标编码，对各点的瓶颈特征进行轻量级修正。
- **物理方程约束 (PINN)**：输出阶段基于网格的拓扑结构，计算 2D 空间声波方程差分，强制引导降噪信号符合物理规律。

这样设计确保了：**网络主要靠信号自身的深度特征来学习何为噪声、何为缺陷回波，而空间传递关系只作为纠偏和辅助。**

---

## 数据准备

与 CAE/PINN 共用 `transformer.py`，数据格式完全相同：

```bash
uv run python transformer.py \
    --noisy /path/to/noisy_21x21.mat \
    --clean /path/to/clean_41x41.mat \
    --output data
```

详见 [instruction.md](instruction.md) 中的数据准备部分。

---

## 训练参数 (scripts/train/train_deepsets_pinn.py)

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | `spatial_cae` | 模型类型：`spatial_cae` (推荐) / `deepsets` (旧) |
| `--base_channels` | `16` | 基础卷积通道数，决定网络容量 (16 对应 ~380K 参数) |
| `--coord_dim` | `64` | 坐标编码维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `data` | 数据目录路径 |
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--dropout` | `0.15` | Dropout 率 |
| `--patience` | `50` | 早停耐心值 |
| `--min_epochs` | `30` | 早停前最小训练轮数 |
| `--no_augment` | `False` | 禁用数据增强 (强烈建议开启增强) |
| `--seed` | `42` | 随机种子 |

### 网格与 Patch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grid_cols` | `41` | 扫描网格列数 |
| `--grid_rows` | `41` | 扫描网格行数 |
| `--patch_size` | `5` | Patch 边长 (5 → 25 点/集合) |
| `--stride` | `1` | Patch 提取步长 |

### 物理约束参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--physics_weight` | `0.0001` | 物理损失权重 λ |
| `--wave_speed` | `5900.0` | 声速 (m/s)，钢材默认 |
| `--center_frequency` | `250000` | 换能器中心频率 (Hz) |
| `--dx` | `0.001` | 网格 x 间距 (m) |
| `--dy` | `0.001` | 网格 y 间距 (m) |

---

## 快速训练命令

### 1. 快速烟雾测试 (2 epoch)

```bash
uv run python scripts/train/train_deepsets_pinn.py --data_path data --epochs 2 --batch_size 32
```

### 2. 基础训练 (推荐)

```bash
uv run python scripts/train/train_deepsets_pinn.py --data_path data --epochs 100 --batch_size 32 --patience 30
```

### 3. 调节物理约束强度

```bash
# 弱物理约束 (几乎纯数据驱动)
uv run python scripts/train/train_deepsets_pinn.py --data_path data --physics_weight 0.0001

# 强物理约束
uv run python scripts/train/train_deepsets_pinn.py --data_path data --physics_weight 0.01
```

### 4. 使用旧版全连接 DeepSetsPINN 模型 (对比用)

```bash
uv run python scripts/train/train_deepsets_pinn.py --data_path data --model_type deepsets --base_channels 16
```

---

## 推理 (scripts/analysis/inference_deepsets.py)

推断脚本会从检查点中自动识别是 `spatial_cae` 还是旧版模型，无需手动指定参数。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input`, `-i` | (必需) | 输入 .mat 文件路径 |
| `--output`, `-o` | `results/deepsets` | 输出目录 |
| `--checkpoint`, `-c` | `checkpoints/best_deepsets_pinn.pth` | 模型检查点 |
| `--cols` | `21` | 输入网格列数 |
| `--rows` | `21` | 输入网格行数 |
| `--target_cols` | `41` | 目标网格列数 |
| `--target_rows` | `41` | 目标网格行数 |
| `--patch_size` | `5` | Patch 边长 |

### 使用示例

```bash
# 基本用法
uv run python scripts/analysis/inference_deepsets.py -i noisy_data.mat -o results/deepsets/

# 指定特定训练的权重
uv run python scripts/analysis/inference_deepsets.py -i noisy.mat -c checkpoints/best_deepsets_pinn.pth
```

---

## 输出文件

### 训练输出

| 文件 | 说明 |
|------|------|
| `fig_deepsets_pinn_training_curves.png` | 训练曲线 (Total/Data/Physics/PSNR) |
| `fig_deepsets_pinn_results.png` | 典型结果：中心信号以及 Patch 的平均降噪效果对比图 |
| `checkpoints/best_deepsets_pinn.pth` | 最佳模型权重 |

### 推理输出

| 文件 | 说明 |
|------|------|
| `deepsets_denoised_YYYYMMDD_HHMMSS.mat` | 41×41 网格插值去噪结果 |
| `deepsets_denoised_YYYYMMDD_HHMMSS_original.mat` | 抽取还原到原始 21×21 网格的结果 |

---

## 调参建议

| 症状 | 故障排查方向 |
|------|------------|
| 训练不收敛或很早就停掉 | `base_channels=32` 可能参数稍大(近10M)，可尝试降低 `--lr 5e-4`，或者增大 `--patience` |
| 结果不如独立 CAE | 可能物理权重太高抢戏了，降低 `--physics_weight 1e-4` |
| 细节变模糊 | `SpatialAuxiliary` 辅助权重可能被放大，验证 `--patch_size` 是否过大，默认为 5 就好 |
