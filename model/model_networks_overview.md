# `model/model.py` 网络结构与训练流程说明

## 概览

`model/model.py` 共定义了 **2 个网络**：

1. `DeepCAE`
2. `DeepCAE_PINN`

其中，`DeepCAE_PINN` 继承自 `DeepCAE`，在相同的编码器-解码器主干上增加了物理残差约束，用于 PINN 训练。

---

## 1. `DeepCAE`

### 1.1 网络结构

`DeepCAE` 是一个 **1D 卷积自编码器**，面向超声一维信号去噪任务，整体采用“**5 层编码器 + 5 层解码器**”的对称结构。

- **输入**：`(B, 1, 1000)`
- **编码端**：5 个 `Conv1d` 下采样块，步长均为 `2`
- **解码端**：5 个 `ConvTranspose1d` 上采样块，逐步恢复长度
- **中间特征**：瓶颈层约为 `(B, 256, 32)`
- **输出**：`(B, 1, 1000)`

#### 编码器

- `enc1`：`Conv1d(1 → 32)` + `BatchNorm1d` + `LeakyReLU`
- `enc2`：`Conv1d(32 → 64)` + `BatchNorm1d` + `LeakyReLU` + `Dropout`
- `enc3`：`Conv1d(64 → 128)` + `BatchNorm1d` + `LeakyReLU` + `Dropout`
- `enc4`：`Conv1d(128 → 256)` + `BatchNorm1d` + `LeakyReLU` + `Dropout`
- `enc5`：`Conv1d(256 → 256)` + `BatchNorm1d` + `LeakyReLU`

#### 解码器

- `dec5`：`ConvTranspose1d(256 → 256)` + `BatchNorm1d` + `LeakyReLU`
- `dec4`：`ConvTranspose1d(256 → 128)` + `BatchNorm1d` + `LeakyReLU` + `Dropout`
- `dec3`：`ConvTranspose1d(128 → 64)` + `BatchNorm1d` + `LeakyReLU` + `Dropout`
- `dec2`：`ConvTranspose1d(64 → 32)` + `BatchNorm1d` + `LeakyReLU`
- `dec1`：`ConvTranspose1d(32 → 1)` + `Tanh`

#### 设计特点

- 使用 `LeakyReLU(0.1)`，增强梯度传播稳定性。
- 采用 `BatchNorm1d` 提升收敛稳定性。
- 在中间层使用 `Dropout`，减轻过拟合。
- 末层使用 `Tanh`，将输出限制在相对平滑的信号范围内。
- 权重初始化使用 `Kaiming Normal`，适合 `LeakyReLU` 激活。

### 1.2 数据训练过程

`DeepCAE` 的训练是标准的**监督式信号重建**流程：

1. **数据准备**
   - 使用 `create_dataloaders(...)` 构建训练集与验证集。
   - 每个样本由 `(noisy, clean)` 构成。
   - `noisy` 作为网络输入，`clean` 作为重建目标。

2. **前向传播**
   - 输入噪声信号 `noisy`，输出去噪结果 `denoised`。

3. **损失函数**
   - 使用 `nn.MSELoss()`。
   - 目标是最小化 `denoised` 与 `clean` 的均方误差。
   - 代码中也以 **提升 PSNR** 作为训练质量指标。

4. **优化方式**
   - 优化器：`Adam`
   - 带 `weight_decay=1e-4` 的 L2 正则
   - 学习率调度：`CosineAnnealingWarmRestarts`

5. **训练监控**
   - 同时记录 `train_loss / val_loss` 与 `train_psnr / val_psnr`
   - 训练前会保存 noisy/clean 样本可视化
   - 训练过程中按验证集 PSNR 保存最优模型，并支持早停


### 1.3 适用场景

适合不引入显式物理约束、直接追求去噪效果的超声信号重建任务。

---

## 2. `DeepCAE_PINN`

### 2.1 网络结构

`DeepCAE_PINN` 继承 `DeepCAE` 的完整卷积自编码器主干，因此其**重建结构与 `DeepCAE` 相同**：

- 同样是 5 层编码 + 5 层解码
- 同样以 `(B, 1, 1000) → (B, 1, 1000)` 为主路径
- 同样输出去噪信号 `denoised`

其差异不在主干，而在于**额外增加了物理残差分支**：

- 通过 `physics_forward(x)` 返回：
  - `denoised`：去噪结果
  - `residual`：物理残差
- 物理残差基于一维时间域阻尼振荡方程近似：
  $$u_{tt} + 2\zeta\omega_0 u_t + \omega_0^2 u = 0$$

### 2.2 物理约束设计

该模型针对单传感器超声时域信号，未直接使用空间二阶导数，而是构造了**时间域 surrogate physics**：

- 使用有限差分计算：
  - 一阶时间导 `u_t`
  - 二阶时间导 `u_tt`
- 用真实采样间隔 `dt = DURATION / (NUM_POINTS - 1)` 进行尺度归一
- 将残差按 `ω0²` 归一化
- 通过局部幅值构造软掩码，仅在有效回波区域加强物理约束

### 2.3 数据训练过程

`DeepCAE_PINN` 的训练是**数据损失 + 物理损失**的联合优化流程：

1. **数据准备**
   - 同样使用 `create_dataloaders(...)`
   - 输入为 `noisy`，监督目标为 `clean`

2. **前向传播**
   - 调用 `physics_forward(noisy)`
   - 同时得到 `denoised` 和 `physics_residual`

3. **损失函数**
   - 数据项：`MSE(denoised, clean)`
   - 物理项：`mean(physics_residual²)`
   - 总损失：`L_total = L_data + λ · L_physics`
   - 其中 `λ` 由 `physics_weight` 控制，默认较小，用于平衡重建与物理一致性

4. **优化方式**
   - 优化器：`Adam`
   - 带 `weight_decay=1e-4`
   - 学习率调度：`CosineAnnealingWarmRestarts`
   - 训练时对梯度进行裁剪，以稳定物理项带来的梯度波动

5. **训练监控**
   - 分别跟踪：
     - `train_total_loss / val_total_loss`
     - `train_data_loss / val_data_loss`
     - `train_physics_loss / val_physics_loss`
     - `train_psnr / val_psnr`
   - 训练前后都会保存样本与结果图
   - 同样支持最优模型保存与早停

### 2.4 适用场景

适合希望在去噪结果中同时保留超声信号物理一致性的场景，尤其是对时域响应规律有约束需求的任务。

---

## 3. 总结

- `model/model.py` 中共包含 **2 个网络**。
- `DeepCAE` 是标准的 1D 卷积自编码器，核心目标是重建 clean 信号。
- `DeepCAE_PINN` 在相同主干上引入物理残差约束，属于物理信息增强版本。
- 两者都采用 **监督学习** 框架；区别主要在于是否加入 **physics loss**。

如果你希望，我也可以继续把这份说明整理成一版更适合 `README` 风格的简版文档，或者直接补充到项目文档体系里。
