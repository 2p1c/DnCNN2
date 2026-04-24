---
name: neural-training-loop
description: 自动训练神经网络（PINN/DeepSets-PINN）直到达到目标指标 val_psnr 或 val_total_loss，支持定时检查checkpoint历史、自动继续训练或停止
---

# Neural Training Loop Skill

自动训练 DnCNN2 神经网络并持续监控直到达到指定指标。适用于本地长时间训练场景。

## 使用方式

```
/neural-training-loop
```

## 项目背景

- **PINN pipeline**: `scripts/train/train.py --pipeline pinn`
- **DeepSets-PINN pipeline**: `scripts/train/train.py --pipeline deepsets`
- **训练输出**: `[TRAIN_METRIC] pipeline=X epoch=N train_total=X val_total=X val_psnr=X ...`
- **Checkpoint 路径**: `results/<run_timestamp>/checkpoints/best_pinn_model.pth` 或 `best_deepsets_pinn.pth`
- **Checkpoint 格式**: `dict` 含 `history` 键，值为 `{"train_total_loss": [...], "val_psnr": [...], ...}`

## 训练指标

| 指标 | 说明 | 优化方向 |
|------|------|---------|
| `val_psnr` | 验证集 PSNR (dB)，越高越好 | `>=` threshold |
| `val_total_loss` | 验证集总损失，越低越好 | `<=` threshold |
| `train_psnr` | 训练集 PSNR | 参考指标 |
| `train_total_loss` | 训练集总损失 | 参考指标 |

## 工作流程

### Step 1: 收集用户输入

使用 `AskUserQuestion` 收集以下参数：

```json
{
  "questions": [
    {
      "header": "训练Pipeline",
      "multiSelect": false,
      "question": "选择要训练的 Pipeline？",
      "options": [
        { "label": "pinn", "description": "DeepCAE + Physics-Informed Neural Network" },
        { "label": "deepsets", "description": "DeepSets-PINN（支持时频融合）" }
      ]
    },
    {
      "header": "训练模式",
      "multiSelect": false,
      "question": "选择训练模式？",
      "options": [
        { "label": "synthetic", "description": "合成数据（num_train/num_val 指定样本数）" },
        { "label": "file", "description": "文件模式（指定 .mat 数据路径）" }
      ]
    },
    {
      "header": "目标指标",
      "multiSelect": false,
      "question": "选择监控的目标指标？",
      "options": [
        { "label": "val_psnr", "description": "验证集 PSNR (dB)，越高越好" },
        { "label": "val_total_loss", "description": "验证集总损失，越低越好" }
      ]
    },
    {
      "header": "目标值",
      "multiSelect": false,
      "question": "输入目标阈值？",
      "options": [
        { "label": "15.0", "description": "基础去噪质量" },
        { "label": "20.0", "description": "较高去噪质量" },
        { "label": "25.0", "description": "高质量去噪（需要更长训练）" },
        { "label": "custom", "description": "手动输入目标值" }
      ]
    },
    {
      "header": "检查间隔",
      "multiSelect": false,
      "question": "选择检查训练状态的时间间隔？",
      "options": [
        { "label": "10m", "description": "每10分钟检查（默认）" },
        { "label": "5m", "description": "每5分钟检查（快速反馈）" },
        { "label": "30m", "description": "每30分钟检查（节省资源）" },
        { "label": "1h", "description": "每小时检查（长时间训练）" },
        { "label": "custom", "description": "手动输入（分钟）" }
      ]
    }
  ]
}
```

**额外参数（通过选项或 custom 输入获取）：**

- `num_epochs_per_batch`: 每次 Cron 检查后训练的 epoch 数（默认 10）
- `data_path`: 文件模式下的 .mat 数据路径（默认 `data`）
- `num_train`: 合成模式训练样本数（默认 5000）
- `num_val`: 合成模式验证样本数（默认 1000）
- `checkpoint_dir`: 指定输出目录（默认 `results/<timestamp>/checkpoints`）

### Step 2: 验证输入

- 如果选择了 `custom` 的目标值，验证输入为有效数字
- 如果选择了 `custom` 间隔，验证在 1-1440 分钟范围内
- 如果选择 `file` 模式但未提供 `data_path`，要求补充

### Step 3: 首次训练启动

立即执行第一次训练（batch_size=配置的 num_epochs_per_batch）：

```bash
uv run python scripts/train/train.py \
  --pipeline {pipeline} \
  --mode {mode} \
  --data_path {data_path} \
  --num_train {num_train} \
  --num_val {num_val} \
  --num_epochs {num_epochs_per_batch} \
  --checkpoint_dir {checkpoint_dir}
```

训练完成后解析 checkpoint 获取最新指标。

### Step 4: 创建 Cron 监控任务

如果首次训练未达标，创建定时任务：

```json
{
  "cron": "*/{interval_minutes} * * * *",
  "prompt": "执行 neural-training-loop 监控任务：\n1. 读取 latest_run.txt 获取最新 checkpoint 路径\n2. 从 checkpoint history 提取最新 val_psnr/val_total_loss\n3. 判断是否达标（val_psnr >= X 或 val_total_loss <= X）\n4. 如果达标：读取 best checkpoint，打印最终结果，删除 latest_run.txt，停止循环\n5. 如果未达标：执行下一轮训练（{num_epochs_per_batch} epochs），更新 checkpoint",
  "durable": true,
  "recurring": true
}
```

### Step 5: 监控逻辑（每次 Cron 执行）

1. **读取最新 checkpoint 路径** — 从 `{checkpoint_dir}/latest_run.txt`（训练脚本会在首次启动时写入）
2. **解析 checkpoint history**:
   ```python
   import torch
   ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
   history = ckpt.get("history", {})
   # 取最后一行的指标
   val_psnr_history = history.get("val_psnr", [])
   val_loss_history = history.get("val_total_loss", [])
   if val_psnr_history:
       current_val_psnr = val_psnr_history[-1]
   if val_loss_history:
       current_val_loss = val_loss_history[-1]
   ```
3. **判断是否达标**:
   - `val_psnr`: `current >= threshold`
   - `val_total_loss`: `current <= threshold`
4. **达标动作**:
   - 打印最终结果表格（epoch, val_psnr, val_loss）
   - 将 `checkpoint_dir` 路径通知用户
   - 使用 `CronDelete` 删除自身任务
5. **未达标动作**:
   - 打印当前进度
   - 执行下一轮训练（追加 epochs）
   - 继续等待

## 指标比较逻辑

| 指标类型 | 比较方式 | 典型阈值 |
|----------|----------|---------|
| val_psnr | `>=` | 15.0 (基础), 20.0 (较高), 25.0 (高质量) |
| val_total_loss | `<=` | 0.1 (好), 0.05 (很好), 0.01 (优秀) |

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| checkpoint 不存在或格式错误 | 等待下一次 Cron 重试；记录错误 |
| 训练脚本执行失败 | 打印错误信息，保留 Cron 任务供手动排查 |
| 历史指标为空 | 跳过该指标的比较，使用另一指标判断 |
| 间隔输入无效 | 要求重新输入有效值 |
| latest_run.txt 丢失 | 搜索 `{checkpoint_dir}/` 下最新的 checkpoint |

## 完整示例

```
用户输入：
- pipeline: deepsets
- mode: file
- data_path: data
- metric: val_psnr
- threshold: 20.0
- interval_minutes: 10
- num_epochs_per_batch: 5

首次启动训练（5 epochs），检查结果：
未达标 → 创建 Cron，每 10 分钟执行：
  1. 加载最新 checkpoint
  2. 检查 val_psnr >= 20.0
  3. 未达标 → 再训练 5 epochs
  4. 重复直到达标或用户手动停止
```

## 注意事项

1. **持久化任务**: 使用 `durable: true` 确保关闭终端后继续执行
2. **自动过期**: Cron 任务 7 天后自动过期；如需更长时间监控可在 CronDelete 前重建
3. **增量训练**: 每次 Cron 执行会在已有 checkpoint 基础上继续训练（使用 `train_from_config` 的 `resume_checkpoint` 逻辑）
4. **路径输出**: 训练开始时将 checkpoint_dir 路径写入 `{checkpoint_dir}/latest_run.txt`
5. **提前停止**: 如果 `early_stopping_patience` 触发但未达标，训练会停止；Cron 会检测到并重新启动
6. **并发保护**: 使用 `{checkpoint_dir}/training.lock` 文件锁防止多个 Cron 同时执行

## Skill 调用

```bash
# 用户执行
/neural-training-loop

# Skill 依次：
1. 询问 pipeline、模式、指标、阈值、间隔
2. 启动首次训练
3. 检查是否达标
4. 未达标则创建 Cron 循环监控
5. 达标后通知用户并清理
```
