# 基于神经网络去噪的非接触式激光超声缺陷检测

> 完整项目代码：[Laser-NDT-Based-NN](https://github.com/2p1c/Laser-NDT-Based-NN.git).

## 项目概述

### 项目背景

非接触式激光超声检测（Laser Ultrasonic Testing, LUT）是一种先进的无损检测技术，利用激光产生和接收超声波信号来检测材料中的缺陷。然而，LUT信号通常受到环境噪声和系统噪声的干扰，导致信噪比（SNR）较低，影响检测的准确性和可靠性。

激光超声依靠热膨胀效应产生超声波，但实际实验中激光参数与环境因素难以完全稳定控制。激光电压、光斑尺寸、材料热扩散特性、激励频率等都会影响信号质量与一致性。

### 核心方法

本项目采用深度学习方法对激光超声信号进行去噪处理，当前主流程提供两种训练 pipeline：

| Pipeline | 特点 | 适用场景 |
|----------|------|----------|
| **PINN** | 基于 DeepCAE 主干 + 物理损失 | 需要保持时域物理一致性 |
| **DeepSets+PINN** | Patch 级空间建模 + 物理约束 | 大面积扫描、复杂空间相关性 |

### 技术亮点

- **Physics-Informed Neural Network (PINN)**：将一维声波方程物理约束嵌入损失函数，确保去噪后的信号符合物理规律
- **DeepSets PINN**：利用 patch/set 空间结构建模邻域关系
- **声学特征验证**：自动对比去噪前后的主频、频带能量、频谱相干性等声学指标
- **统一推理管线**：支持训练与推理使用不同网格尺寸（灵活应对实际检测场景）

---

## 快速开始

### 环境配置

```bash
# 安装依赖
uv sync

# 可选：安装开发依赖
uv sync --group dev
```

### 一键运行完整流程

```bash
# 使用统一管线（transform → train → inference → validation）
uv run python scripts/run_unified_pipeline.py \
    --config configs/pipeline_pinn_template.json
```

### 分步运行

#### 1. 数据转换

将实验 `.mat` 数据转换为训练格式：

```bash
uv run python scripts/transformer.py \
    --noisy data/noisy_21x21.mat \
    --clean data/clean_41x41.mat \
    --output data
```

#### 2. 模型训练

```bash
# PINN (推荐)
uv run python scripts/train/train.py --pipeline pinn --mode file --data_path data

# DeepSets + PINN
uv run python scripts/train/train.py --pipeline deepsets --mode file --data_path data
```

#### 3. 推理去噪

```bash
# CAE / PINN 推理
uv run python scripts/analysis/inference.py \
    --input noisy.mat \
    --output results/

# DeepSets 推理
uv run python scripts/analysis/inference_deepsets.py \
    --input noisy.mat \
    --output results/
```

---

## 核心参数

### 信号参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 采样率 | 6.25 MHz | |
| 信号时长 | 160 μs | |
| 采样点数 | 1000 | |
| 中心频率 | 250 kHz | |

### 网格参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 输入网格 | 21×21 | 噪声信号网格尺寸 |
| 目标网格 | 41×41 | 干净信号/输出网格尺寸 |

### 物理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 声速 | 5900 m/s | 钢中典型值 |
| 阻尼比 | 0.05 | PINN 物理损失参数 |

---

## 输出目录结构

```
DnCNN2/
├── results/
│   ├── checkpoints/     # 模型权重
│   │   ├── best_pinn_model.pth
│   │   └── best_deepsets_pinn.pth
│   ├── images/          # 可视化结果
│   └── *.mat           # 推理输出
├── data/               # 转换后的训练数据
│   ├── train/noisy/
│   ├── train/clean/
│   ├── val/noisy/
│   └── val/clean/
└── experiments/        # 实验记录
```

---

## 统一管线配置

推荐使用 JSON 配置文件运行统一管线：

```bash
uv run python scripts/run_unified_pipeline.py \
    --config configs/pipeline_pinn_template.json
```

配置文件示例（`configs/pipeline_pinn_template.json`）：

```json
{
  "pipeline": "pinn",
  "noisy_mat": "data/noisy.mat",
  "clean_mat": "data/clean.mat",
  "inference_input": "data/noisy.mat",

  "input_cols": 21,
  "input_rows": 21,
  "target_cols": 41,
  "target_rows": 41,

  "inference_input_cols": 29,
  "inference_input_rows": 29,
  "inference_target_cols": 57,
  "inference_target_rows": 57,

  "epochs": 50,
  "batch_size": 32,
  "lr": 0.001,
  "physics_weight": 0.001
}
```

### 配置参数说明

| 参数 | 说明 |
|------|------|
| `pipeline` | 选择训练管线：`pinn` 或 `deepsets` |
| `noisy_mat` | 噪声数据路径（transform 阶段） |
| `clean_mat` | 干净数据路径（transform 阶段） |
| `inference_input` | 推理输入数据路径 |
| `input_cols/rows` | 训练输入网格尺寸 |
| `target_cols/rows` | 训练目标网格尺寸 |
| `inference_*_cols/rows` | 推理网格尺寸（可与训练不同） |
| `skip_transform` | 跳过数据转换，复用已有 data/ |
| `skip_train` | 跳过训练，直接推理 |

---

## 推理网格与训练网格分离

训练和推理可以使用不同的网格尺寸，这是本项目的重要特性：

- **训练网格**：由 `input_cols/input_rows/target_cols/target_rows` 指定
- **推理网格**：由 `inference_input_cols/inference_input_rows/inference_target_cols/inference_target_rows` 指定

这使得你可以：
- 用小面积标注数据训练模型
- 对大面积扫描区域进行推理

---

## 声学验证

推理过程中会自动进行声学特征验证，对比去噪前后的：

- 主频 (Dominant Frequency)
- 子频带能量 (Sub-band Energy)
- 频谱相干性 (Spectral Coherence)
- 峰值振幅 (Peak Amplitude)

验证结果保存为图像文件，并输出终端报告。

---

## 调参建议

### 训练问题与对策

| 问题 | 解决方案 |
|------|----------|
| 验证 PSNR 不升反降 | 降低学习率至 `5e-4` 或 `1e-4` |
| 训练不稳定/震荡 | 减小 `physics_weight`，或减小 `batch_size` |
| 去噪很强但信号形态变差 | 适度减小 `physics_weight`、减小 `dropout` |
| 高频细节丢失 | DeepSets 里尝试 `patch_size=7`，提高 `epochs` |
| 小振幅信号插值异常 | `interp_method` 改为 `linear` |

### 推荐配置组合

**快速验证（先跑通）**：
```json
{
  "epochs": 20,
  "batch_size": 16,
  "lr": 0.001
}
```

**稳健训练（推荐）**：
```json
{
  "epochs": 80,
  "batch_size": 32,
  "lr": 0.0005,
  "patience": 50,
  "min_epochs": 30
}
```

**强物理约束**：
```json
{
  "physics_weight": 0.005,
  "epochs": 100,
  "lr": 0.0005
}
```

---

## 相关文档

- [配置文件说明](../configs/README.md)
- [模型结构说明](../model/model_networks_overview.md)

---

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
