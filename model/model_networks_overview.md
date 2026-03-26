# `model/model.py` 网络结构说明（重构后）

## 1. 文件定位与重构结果

当前项目的模型定义已收敛到单一文件：`model/model.py`。

本次重构后，模型层只保留 3 个核心模型类：

1. `DeepCAE`
2. `DeepCAE_PINN`
3. `DeepSetsPINN`

并且有如下约束：

- `DeepCAE_PINN` 继承自 `DeepCAE`。
- `SpatialAuxiliaryCAE` 已删除。
- 原先的重命名别名（如 `SpatialContextCAE`、`SetInvariantWavePINN`）不再作为独立模型入口。
- `model/model_deepsets.py` 已移除，不再维护双模型文件结构。

---

## 2. `DeepCAE`

### 2.1 任务定位

`DeepCAE` 是 1D 超声信号去噪主干网络，适用于标准的监督式重建。

### 2.2 结构概览

- 输入：`(B, 1, 1000)`
- 编码：5 层 `Conv1d(stride=2)`
- 解码：5 层 `ConvTranspose1d(stride=2)`
- 输出：`(B, 1, 1000)`

编码阶段逐步提取时序特征，解码阶段恢复信号长度与波形细节。

### 2.3 关键实现点

- 激活函数：`LeakyReLU(0.1)`
- 归一化：`BatchNorm1d`
- 正则化：中间层 `Dropout`
- 输出层：`Tanh`

这套设计在去噪任务中可保持较好的收敛稳定性和重建平滑性。

---

## 3. `DeepCAE_PINN`

### 3.1 任务定位

`DeepCAE_PINN` 在 `DeepCAE` 主干上叠加物理约束，用于 PINN 训练路线。

### 3.2 继承关系

- 继承：`DeepCAE_PINN(DeepCAE)`
- 普通推理：沿用 `DeepCAE.forward(...)`
- 训练专用：`physics_forward(...) -> (denoised, residual)`

### 3.3 物理残差形式

使用时间域阻尼振子近似残差：

`r = (u_tt + 2*zeta*omega0*u_t + omega0^2*u) / omega0^2`

其中：

- `u_t`, `u_tt` 使用有限差分近似
- `dt = DURATION / (NUM_POINTS - 1)`
- 使用局部能量软掩码抑制静默区对物理损失的干扰

### 3.4 训练损失

训练侧采用：

`L_total = L_data + lambda * L_physics`

- `L_data`：重建误差（MSE）
- `L_physics`：残差平方均值
- `lambda`：`physics_weight`

---

## 4. `DeepSetsPINN`

### 4.1 任务定位

`DeepSetsPINN` 用于 patch/set 形式的二维扫描点去噪，并结合波动方程残差约束。

该模型承接原 DeepSets 路线，现在作为唯一 DeepSets 训练与推理模型入口。

### 4.2 网络组成

`DeepSetsPINN` 由 4 个子模块组成：

1. `SignalEncoder`：逐点信号编码（1D Conv）
2. `CoordinateMLP`：坐标嵌入（x, y）
3. `PointEncoder`：融合信号嵌入与坐标嵌入
4. `SignalDecoder`：逐点重建输出信号

前向流程（简化）：

- 输入：`noisy_signals (B, R, T)` 与 `coordinates (B, R, 2)`
- 输出：`denoised (B, R, T)`

其中 `R = patch_size^2`。

### 4.3 物理残差

`compute_wave_equation_residual(...)` 基于 patch 内部点近似二维波动方程：

- 计算 `u_tt`, `u_xx`, `u_yy`
- 构造归一化残差：`(u_tt - c^2*(u_xx + u_yy)) / omega0^2`
- 使用软掩码对高能区加强约束

训练接口通过 `physics_forward(...)` 同时返回重建结果与残差。

---

## 5. 与训练脚本的对应关系

重构后训练入口统一为：`scripts/train/train.py`。

- `pipeline = "pinn"`：调用 `train_pinn(...)`，内部使用 `DeepCAE_PINN`
- `pipeline = "deepsets"`：调用 `train_deepsets_pinn(...)`，内部使用 `DeepSetsPINN`

`scripts/run_unified_pipeline.py` 不再直接引用旧的双训练脚本，而是通过
`train_from_config(...)` 路由到对应训练分支。

---

## 6. 当前架构原则

本项目模型层遵循以下简化原则：

1. 单文件维护：统一在 `model/model.py` 管理核心模型。
2. 单一入口：每条训练路径只保留一个有效模型实现。
3. 去除冗余别名：避免历史命名并存导致的配置歧义。
4. 保持可运行闭环：保证 `scripts/run_unified_pipeline.py` 可直接贯通 transform/train/inference。

---

## 7. 后续维护建议

如需新增模型，建议遵循：

- 明确新增模型服务于哪条 pipeline（`pinn` 或 `deepsets`）
- 同步更新 `scripts/train/train.py` 的配置路由
- 同步更新对应 inference 的 checkpoint 反序列化逻辑
- 避免新增仅用于兼容历史命名的重复类

这样可以持续保持当前重构后的结构简洁性和可维护性。
