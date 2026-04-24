# Training GUI 使用说明

## 功能概览

`train_gui.py` 提供 Unified Pipeline 的图形化入口，支持：

- 完整流程：`transform + train + inference`
- 仅训练（勾选 `skip_inference`）
- 仅推理（勾选 `skip_train`）
- 仅数据转换（勾选 `skip_train` 与 `skip_inference`）
- 续训（填写 `resume_checkpoint` 或点击“选择续训Checkpoint”）
- 单任务运行与排队/终止
- 实时曲线（Total/Data/Physics Loss、PSNR、LR）
- 每次运行自动保存 `run.log`、`metrics.jsonl`、`hparams_snapshot.json`
- 从日志回放训练曲线、恢复超参数、查看结果图

## 启动方式

在仓库根目录执行：

```bash
uv run python scripts/train/train_gui.py
```

## 日志与配置落盘

每次运行会在 `results/<timestamp>/` 下生成：

- `logs/run.log`
- `logs/metrics.jsonl`
- `logs/hparams_snapshot.json`
- `logs/launch_command.txt`

运行输出图通常位于 `results/<timestamp>/images/`。

## 配置保存/加载

GUI 左侧支持保存和加载参数配置，默认保存目录：

- `scripts/train/configs/`

## 注意事项

- 当 `skip_inference=false` 时，`inference_input` 必须提供。
- 若只做推理，需同时设置 `skip_train=true` 并提供可用 `checkpoint`。
- 若只做训练，设置 `skip_inference=true`。
- 续训时，`resume_checkpoint` 必须与当前模型分支匹配（PINN/DeepSets/TF-Fusion）。
