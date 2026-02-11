# CAE 超声信号去噪训练指南

## 数据准备 (transformer.py)

将实验 .mat 数据转换为训练格式。

### 基本用法

```bash
uv run python transformer.py --noisy /Volumes/ESD-ISO/数据/260129/jiguang/21_21.mat --clean /Volumes/ESD-ISO/数据/260129/yadian250k/41_41.mat
```

### 信号长度裁剪

如果采集的信号超过 1000 个点（如 1500 点），使用 `--signal_length` 参数自动截取前 N 个点：

```bash
# 默认截取前 1000 个点
uv run python transformer.py --noisy noisy.mat --clean clean.mat

# 自定义截取长度（如 800 个点）
uv run python transformer.py --noisy noisy.mat --clean clean.mat --signal_length 800
```

### transformer.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--noisy` | (必需) | 噪声信号 .mat 文件路径 |
| `--clean` | (必需) | 干净信号 .mat 文件路径 |
| `--output` | `data` | 输出目录 |
| `--noisy_cols` | `21` | 噪声数据网格列数 |
| `--noisy_rows` | `21` | 噪声数据网格行数 |
| `--clean_cols` | `41` | 干净数据网格列数 |
| `--clean_rows` | `41` | 干净数据网格行数 |
| `--augment_factor` | `5` | 数据增强倍数 (1=不增强) |
| `--train_ratio` | `0.8` | 训练集比例 |
| `--interp_method` | `cubic` | 插值方法：`linear` / `cubic` |
| `--signal_length` | `1000` | 目标信号长度 (超过则截断) |
| `--no_normalize` | `False` | 禁用信号归一化 |
| `--seed` | `42` | 随机种子 |

---

## 训练参数 (train.py)

| 参数           | 默认值        | 说明                                                       |
| -------------- | ------------- | ---------------------------------------------------------- |
| `--mode`       | `synthetic`   | 数据模式：`synthetic`(合成) / `file`(文件)                 |
| `--data_path`  | `data`        | 数据目录路径 (file模式)                                    |
| `--model`      | `lightweight` | 模型类型：`lightweight`(3层) / `deeper`(4层) / `deep`(5层) |
| `--epochs`     | `50`          | 训练轮数                                                   |
| `--batch_size` | `32`          | 批次大小                                                   |
| `--lr`         | `0.001`       | 学习率                                                     |
| `--dropout`    | `0.1`         | Dropout率 (推荐0.1-0.2)                                    |
| `--patience`   | `50`          | 早停耐心值                                                 |
| `--min_epochs` | `30`          | 早停前最小训练轮数                                         |
| `--augment`    | `False`       | 启用数据增强                                               |
| `--num_train`  | `5000`        | 合成训练样本数                                             |
| `--num_val`    | `1000`        | 合成验证样本数                                             |
| `--seed`       | `42`          | 随机种子                                                   |

---

## 快速训练命令

### 1. 合成数据 - 快速测试

```bash
python train.py --model lightweight --epochs 30
```

### 2. 合成数据 - 完整训练

```bash
python train.py --model deep --epochs 100 --dropout 0.2
```

### 3. 实验数据 - 基础训练

```bash
python train.py --mode file --data_path ./data --model deep --epochs 100
```

### 4. 实验数据 - 防过拟合 (推荐)

```bash
python train.py --mode file --data_path ./data --model deep --epochs 100 --dropout 0.2 --patience 20
```

### 5. 实验数据 - 数据增强

```bash
python train.py --mode file --data_path ./data --model deep --epochs 150 --dropout 0.1 --augment
```

---

## 输出文件

| 文件                         | 说明                 |
| ---------------------------- | -------------------- |
| `fig_pre_train_samples.png`  | 训练前样本可视化     |
| `fig_results.png`            | 去噪效果对比图       |
| `fig_training_curves.png`    | 训练曲线 (Loss/PSNR) |
| `checkpoints/best_model.pth` | 最佳模型权重         |

---

## 调参建议

| 问题                  | 解决方案                                  |
| --------------------- | ----------------------------------------- |
| 过拟合 (Train好Val差) | 增加`--dropout`到0.2-0.3，启用`--augment` |
| 欠拟合 (两者都差)     | 使用`--model deep`，降低`--dropout`       |
| 训练不稳定            | 降低`--lr`到0.0005                        |
| 早停太早              | 增加`--patience`，增加`--min_epochs`      |

---

## 推理 (inference.py)

使用训练好的模型对新数据进行去噪。

### 命令行参数

| 参数                 | 默认值                       | 说明                         |
| -------------------- | ---------------------------- | ---------------------------- |
| `--input`, `-i`      | (必需)                       | 输入 .mat 文件路径           |
| `--output`, `-o`     | `results/denoised`           | 输出目录                     |
| `--checkpoint`, `-c` | `checkpoints/best_model.pth` | 模型检查点路径               |
| `--model`, `-m`      | `deep`                       | 模型类型                     |
| `--cols`             | `21`                         | 输入网格列数                 |
| `--rows`             | `21`                         | 输入网格行数                 |
| `--target_cols`      | `41`                         | 目标网格列数                 |
| `--target_rows`      | `41`                         | 目标网格行数                 |
| `--interp_method`    | `cubic`                      | 插值方法：`linear` / `cubic` |
| `--batch_size`       | `64`                         | 推理批次大小                 |
| `--signal_length`    | `1000`                       | 目标信号长度 (超过则截断)    |
| `--no_original_size` | `False`                      | 不保存原始尺寸结果           |

### 使用示例

```bash
# 基本用法
python inference.py --input noisy_data.mat --output results/denoised/

# 指定模型和检查点
python inference.py -i noisy.mat -o results/ -c checkpoints/best_model.pth -m deep

# 自定义网格尺寸
python inference.py --input data.mat --cols 21 --rows 21 --target_cols 41 --target_rows 41

# 只保存完整分辨率结果
python inference.py --input data.mat --output results/ --no_original_size
```

### 输出文件

| 文件                                    | 说明                        |
| --------------------------------------- | --------------------------- |
| `denoised_YYYYMMDD_HHMMSS_full.mat`     | 插值后的 41×41 网格去噪结果 |
| `denoised_YYYYMMDD_HHMMSS_original.mat` | 还原到原始 21×21 网格的结果 |

### .mat 文件格式

**输入文件要求**：

- `x`: 时间向量
- `y`: 信号数据 (n_points × signal_length)

**输出文件包含**：

- `x`: 时间向量
- `y`: 去噪后的信号数据
- `signal_length`: 信号长度
- `grid_cols`, `grid_rows`: 网格尺寸
- `n_signals`: 信号数量
