修复 DeepSets PINN 训练问题
问题诊断
从训练结果可以看到两个问题：

问题 1：模型输出坍塌为近零信号
从 
fig_deepsets_pinn_results.png
 可以清楚看到绿色 "Denoised" 线几乎为零。 PSNR 从 epoch 1 就到 12.4 dB 之后不再变化 — 这就是直接预测零的 PSNR。

根因：当前架构要求模型从 feature vector 完全重建 1000-point 信号：

信号 → Conv1d Encoder → 256-d → PointEncoder → MeanPool → Concat → Linear → ConvTranspose1d × 5 → 1000-point
这条路径太长，信息严重损失。原始 DeepCAE 能工作是因为 encoder-decoder 直接端到端处理同一个信号，而 DeepSets 在中间做了 set aggregation，cut 掉了 per-signal 捷径。

问题 2：训练速度慢（~25s/epoch）
每个 batch 需要对 B×R 个信号做 Conv1d encoder + ConvTranspose1d decoder（B=32, R=9 → 288 次 conv 前向），开销很大。

修复方案
IMPORTANT

核心思路：从 "重建信号" 切换到 "预测残差（噪声）"。这样模型只需学习一个小的修正量，不需要完全重建信号。

改动 1 — 残差学习（Residual Learning）
修改 
forward()
 使其 预测噪声，最终输出 = input − predicted_noise：

diff
# 当前：模型直接输出 denoised signal
-denoised = decoder(concat(point_feat, global_feat))
+# 改为：模型预测噪声 residual
+noise = decoder(concat(point_feat, global_feat))
+denoised = noisy_signals - noise
好处：

模型只需学习噪声分量（通常比信号简单）
输入信号通过 skip connection 直接传递到输出
与 DnCNN 等经典去噪网络的设计一致
改动 2 — 简化解码器
当前 5 层 ConvTranspose1d 解码器对残差预测来说过于复杂。改为更轻量的 3 层解码器：

Linear → (128, 32) → ConvTranspose1d×3 → (1, 1000) → 无 Tanh
移除 Tanh（残差可以是任意值，不限制在 [-1,1]）。

改动 3 — 简化编码器
5 层 Conv1d 编码器同样过重。改为 3 层，减少计算量：

Conv1d: 1 → 16 → 32 → 64, stride=4, kernel=7
AdaptiveAvgPool1d(1) → Linear → 128-d
Proposed Changes
DeepSets Model
[MODIFY] 
model_deepsets.py
SignalEncoder: 5 层 → 3 层，stride=4 提升下采样速度
SignalDecoder: 5 层 → 3 层，移除 Tanh 激活
DeepSetsPINN.forward(): 添加残差学习 denoised = input - noise
更新 
init
 中 base_channels 默认为 16
Verification Plan
Automated Tests
bash
# 1. 形状测试
uv run python model_deepsets.py
# 2. 导入链测试
uv run python -c "from train_deepsets_pinn import *; print('OK')"
# 3. 快速训练烟雾测试（2 epoch）— 确认 PSNR 能提升
uv run python train_deepsets_pinn.py --data_path data --epochs 2 --batch_size 32
预期结果
模型参数 < 500K
每 epoch < 15 秒
PSNR 在前几个 epoch 应该有明显提升（不再卡在 12.4 dB）

Comment
⌥⌘M
