# Unified Pipeline Config Quick Reference

本页用于快速配置 `scripts/run_unified_pipeline.py`。  
推荐先复制模板再改：

- `configs/pipeline_pinn_template.json`
- `configs/pipeline_deepsets_template.json`

---

## 1) 最小可运行步骤

1. 准备 `.mat` 文件：
   - `noisy_mat`：噪声输入（通常 21x21）
   - `clean_mat`：干净标签（通常 41x41）
   - `inference_input`：推理输入（通常可与 noisy_mat 相同）
2. 选择管线：`pipeline = "pinn"` 或 `"deepsets"`
3. 先用模板默认值跑通一版
4. 再调超参数

运行方式：

```bash
uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json
```

---

## 2) 参数分组速查

### A. 核心流程控制

| 参数             | 作用              | 推荐                             |
| ---------------- | ----------------- | -------------------------------- |
| `pipeline`       | 选择训练/推理分支 | `pinn` 或 `deepsets`             |
| `skip_transform` | 跳过数据转换      | 复用已有 `data/` 时设 `true`     |
| `skip_train`     | 跳过训练直接推理  | 复用已有 checkpoint 时设 `true`  |
| `checkpoint`     | 指定模型权重文件  | 不填则自动用默认 best checkpoint |

### B. 输入输出路径

| 参数              | 作用             | 说明                          |
| ----------------- | ---------------- | ----------------------------- |
| `noisy_mat`       | 转换阶段噪声数据 | `skip_transform=false` 时必填 |
| `clean_mat`       | 转换阶段干净数据 | `skip_transform=false` 时必填 |
| `inference_input` | 推理输入数据     | 始终必填                      |
| `data_dir`        | 训练数据目录     | 默认 `data`                   |
| `results_dir`     | 输出目录         | 默认 `results`                |

### C. 网格与信号参数

| 参数                         | 作用             | 推荐默认                         |
| ---------------------------- | ---------------- | -------------------------------- |
| `input_cols`, `input_rows`   | 输入网格大小     | `21, 21`                         |
| `target_cols`, `target_rows` | 目标网格大小     | `41, 41`                         |
| `signal_length`              | 信号长度（截断） | `1000`                           |
| `interp_method`              | 空间插值         | `cubic`（极小幅值可改 `linear`） |

### D. 数据构建参数（transform）

| 参数             | 作用       | 推荐            |
| ---------------- | ---------- | --------------- |
| `augment_factor` | 扩增倍数   | `3~8`（默认 5） |
| `train_ratio`    | 训练集比例 | `0.8`           |

### E. 训练参数（共用）

| 参数         | 作用         | 推荐起点            |
| ------------ | ------------ | ------------------- |
| `epochs`     | 最大轮数     | `50~150`            |
| `batch_size` | 批大小       | `16~64`（显存决定） |
| `lr`         | 学习率       | `1e-3` 起步         |
| `dropout`    | Dropout 比例 | `0.1~0.2`           |
| `seed`       | 随机种子     | `42` 固定方便复现   |
| `patience`   | 早停耐心     | `30~80`             |
| `min_epochs` | 最小训练轮数 | `20~40`             |

### F. 物理参数（PINN核心）

| 参数               | 作用           | 推荐                           |
| ------------------ | -------------- | ------------------------------ |
| `physics_weight`   | 物理损失权重 λ | PINN: `1e-3`，DeepSets: `1e-4` |
| `wave_speed`       | 声速 (m/s)     | 钢中常用 `5900`                |
| `center_frequency` | 中心频率 (Hz)  | `250000`                       |
| `damping_ratio`    | 阻尼比（PINN） | `0.03~0.08`（默认 0.05）       |

### G. DeepSets 专有参数

| 参数            | 作用                  | 推荐              |
| --------------- | --------------------- | ----------------- |
| `patch_size`    | patch 边长            | `5`（常用）       |
| `stride`        | patch 步长            | `1`（质量优先）   |
| `model_type`    | DeepSets 分支模型开关 | 固定为 `deepsets` |
| `base_channels` | 基础通道数            | `16` 起步         |
| `coord_dim`     | 坐标嵌入维度          | `64`              |

#### DeepSets TF-Fusion 扩展参数

当 `model_type=tf_fusion` 时，transform 阶段会额外生成 `train/tf` 和 `val/tf`。

| 参数 | 作用 | 推荐 |
|---|---|---|
| `signal_embed_dim` | 时域分支嵌入维度 | `128` |
| `coord_embed_dim` | 坐标分支嵌入维度 | `64` |
| `point_dim` | 点特征维度 | `128` |
| `tf_embed_dim` | 时频分支嵌入维度 | 与 `signal_embed_dim` 相同 |
| `stft_n_fft` | STFT FFT 长度 | `128` |
| `stft_hop_length` | STFT 帧移 | `32` |
| `stft_win_length` | STFT 窗长 | `128` |
| `stft_window` | STFT 窗函数 | `hann` |
| `stft_pooling` | 频轴压缩方式 | `mean` |
| `fusion_mode` | 时域/时频融合方式 | `gated`（主方法）/`concat`（消融） |
| `debug_numerics` | 启用 NaN/Inf 检查 | `false` |

注意：训练与推理的 STFT 参数必须一致。`inference_deepsets.py` 在 `tf_fusion` 模式会和 checkpoint 参数严格比对，不一致会报错退出。

### H. 推理与验证

| 参数                   | 作用           | 推荐     |
| ---------------------- | -------------- | -------- |
| `inference_batch_size` | 推理批大小     | `32~128` |
| `validation_samples`   | 声学验证采样数 | `20~100` |

---

## 3) 调参优先级（建议顺序）

1. **先固定数据与网格**：`signal_length`、grid、`interp_method`
2. **再调收敛稳定性**：`lr`、`batch_size`、`epochs`、`patience`
3. **再调物理约束强度**：`physics_weight`
4. **最后调模型容量**：`dropout`、`base_channels`、`patch_size`

---

## 4) 常见现象与对策

- 验证 PSNR 不升反降：先把 `lr` 降到 `5e-4` 或 `1e-4`
- 训练不稳定/震荡：减小 `physics_weight`，或减小 `batch_size`
- 去噪很强但信号形态变差：适度减小 `physics_weight`、减小 `dropout`
- 高频细节丢失：DeepSets 里尝试 `patch_size=7`，并提高 `epochs`
- 小振幅信号插值异常：`interp_method` 改为 `linear`

---

## 5) 常用配置组合

### 快速验证（先跑通）

```json
{
  "epochs": 20,
  "batch_size": 16,
  "lr": 0.001,
  "validation_samples": 20
}
```

### 稳健训练（推荐）

```json
{
  "epochs": 80,
  "batch_size": 32,
  "lr": 0.0005,
  "patience": 50,
  "min_epochs": 30
}
```

### 强物理约束（需谨慎）

```json
{
  "physics_weight": 0.005,
  "epochs": 100,
  "lr": 0.0005
}
```

---

## 6) 实用命令

- **使用模板运行，启用日志记录功能**：

```bash

uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json --log_experiment

```

- **用模板 + 临时覆盖参数**：

```bash

uv run python scripts/run_unified_pipeline.py --config configs/pipeline_deepsets_template.json --epochs 100 --lr 5e-4

```

- **跳过训练只做推理+验证**：

```bash

uv run python scripts/run_unified_pipeline.py --config configs/pipeline_pinn_template.json --skip_train

```

## 注：布尔参数推荐在 JSON 中改（`true/false`）。命令行上 `--skip_train` 出现即为开启。

# 云服务器运行指令

```bash
source .venv/bin/activate
python -m scripts.run_unified_pipeline --config configs/pipeline_pinn.json
```
