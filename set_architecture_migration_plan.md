# 从当前 Conv1d 自编码器迁移到 DeepSets 或 Set Transformer 的方案

## 1. 当前基线结构

当前模型本质上是单接收点 1D 卷积自编码器：

```text
输入 noisy signal
shape: (B, 1, T)
   |
   v
Conv1d Encoder
   |
   v
latent feature
   |
   v
ConvTranspose1d Decoder
   |
   v
输出 denoised signal
shape: (B, 1, T)
```

其中：

- `B` 是 batch size
- `T` 是时间采样点数，当前为 1000

这套结构隐含一个前提：每个样本只有一个接收点，或者多个接收点已经被提前压缩成固定维度输入。

## 2. 迁移目标

迁移后的核心目标不是单纯替换网络，而是改掉输入数据的建模方式。

希望支持的能力包括：

- 接收点数量 `R` 可变化
- 接收点位置不要求规则网格
- 接收点顺序不应影响结果
- 未来可以更自然地加入空间相关物理约束

## 3. 迁移前后的输入输出视角变化

### 当前 Conv1d 自编码器

```text
单接收点输入:

sample
  -> signal(t)

张量形式:
  x: (B, 1, T)
  y: (B, 1, T)
```

### 迁移后数据概念

单个样本将不再只是“一条信号”，而是一组接收点观测：

$$
\{(x_i, s_i(t))\}_{i=1}^{R}
$$

其中：

- `x_i` 是第 `i` 个接收点位置
- `s_i(t)` 是该接收点的一条时间序列
- `R` 是该样本的接收点数量，可变

## 4. 迁移到 DeepSets 的结构图

### 4.1 输入输出结构

```text
一个样本:
{ (x1, s1(t)), (x2, s2(t)), ..., (xR, sR(t)) }

每个点单独编码:
(xi, si(t))
   |
   v
Point Encoder phi
   |
   v
hi

所有点聚合:
{h1, h2, ..., hR}
   |
   v
sum / mean / attention pooling
   |
   v
global representation g

输出头:
g -> global prediction
or
(hi, g) -> per-point denoised signal
```

### 4.2 推荐的具体实现拆分

建议把每个接收点输入拆成两部分：

1. 时间信号编码器
2. 坐标编码器

结构如下：

```text
si(t) shape: (T,)
   |
   v
1D Signal Encoder
   |
   v
signal embedding es_i

xi shape: (Dx,)
   |
   v
Coordinate MLP
   |
   v
position embedding ex_i

concat(es_i, ex_i)
   |
   v
Point Fusion MLP
   |
   v
hi
```

再对所有 `hi` 做集合聚合：

```text
h1, h2, ..., hR
   |
   v
mean / sum pooling
   |
   v
g
```

最后有两种输出路径。

#### 路径 A：输出整个样本的全局表征

适用于分类、回归、缺陷打分等任务：

```text
g -> MLP -> output
```

#### 路径 B：输出每个接收点的去噪信号

更贴近当前任务：

```text
for each point i:
concat(hi, g)
   |
   v
Point Decoder
   |
   v
denoised si(t)
```

输出可以组织为：

- 列表形式：每个样本输出 `R` 条时间序列
- padding 形式：`(B, R_max, T)` 外加有效 mask

### 4.3 DeepSets 方案的优点

- 最容易从当前项目起步
- 不强依赖固定 `R`
- 支持缺失传感器
- 架构相对简单，训练稳定性更好

### 4.4 DeepSets 方案的风险

- 点间关系建模较弱
- 如果传播模式复杂，可能利用不够充分

## 5. 迁移到 Set Transformer 的结构图

### 5.1 输入输出结构

```text
一个样本:
{ (x1, s1(t)), (x2, s2(t)), ..., (xR, sR(t)) }
   |
   v
Per-point encoder
   |
   v
tokens: [h1, h2, ..., hR]
   |
   v
Set Transformer blocks
   |
   v
context-aware tokens: [h1', h2', ..., hR']
   |
   +--> pooled global representation g
   |
   +--> per-point decoder for each receiver
```

### 5.2 推荐的模块划分

第一步和 DeepSets 相同，先把每个接收点编码成 token：

```text
(xi, si(t)) -> Point Encoder -> hi
```

然后把这些 token 送入集合注意力模块：

```text
[h1, h2, ..., hR]
   |
   v
Self-Attention over set
   |
   v
[h1', h2', ..., hR']
```

这样每个 `hi'` 不再只表示单个接收点本身，而是已经吸收了其他接收点的信息。

最后也有两种输出路径。

#### 路径 A：集合级输出

```text
[h1', ..., hR']
   |
   v
Pooling / PMA
   |
   v
g
   |
   v
MLP head
```

#### 路径 B：逐点去噪输出

```text
for each point i:
hi'
   |
   v
Point Decoder
   |
   v
denoised si(t)
```

### 5.3 Set Transformer 方案的优点

- 更强地建模点间依赖
- 更适合传播时延、相位关系、反射模式等任务
- 更适合不规则空间布局

### 5.4 Set Transformer 方案的风险

- 实现更复杂
- 训练更慢
- 更吃数据量和调参经验

## 6. 输入张量应该怎么组织

为了兼容可变 `R`，推荐不要把数据集的核心表示写死成固定 `(R, T)`，而是先用更语义化的样本结构。

### 推荐的单样本结构

```text
sample = {
  receiver_positions: (R, Dx),
  noisy_signals: (R, T),
  clean_signals: (R, T),
  receiver_mask: (R,)   # 可选
}
```

其中：

- `Dx` 是坐标维度，1D 场景可取 1，若后续扩展到 2D/3D 可取 2 或 3
- `receiver_mask` 用于 batch padding 后标记有效接收点

### 推荐的 batch 组织方式

如果一个 batch 中不同样本的 `R` 不同，可以在 collate 时 pad 到 `R_max`：

```text
receiver_positions: (B, R_max, Dx)
noisy_signals:      (B, R_max, T)
clean_signals:      (B, R_max, T)
receiver_mask:      (B, R_max)
```

注意：

- 模型语义上处理的是集合
- padding 只是实现层面的打包手段
- 有效点由 `receiver_mask` 控制

## 7. 与当前损失函数的关系

### 监督去噪损失

迁移后仍然可以保留当前的重建损失：

$$
L_{data} = \text{MSE}(\hat{s}_i(t), s_i^{clean}(t))
$$

如果是逐点输出，就对所有有效接收点求平均。

### 物理损失的升级方向

一旦有多个接收点和位置坐标，就可以考虑比当前更接近真实传播约束的物理损失，例如：

#### 方案 1：接收点内的时间残差

继续沿用每个点各自的时间域残差：

$$
r_t = u_{tt} + 2\zeta\omega_0 u_t + \omega_0^2 u
$$

#### 方案 2：接收点间的空间差分残差

如果接收点近似沿一条线分布，且位置足够规则，可加入：

$$
r_{pde} = u_{tt} - c^2 u_{xx}
$$

其中 `u_xx` 可以由接收点空间邻域差分得到。

#### 方案 3：基于到达时间的几何一致性

如果空间点分布不规则，不一定强行做规则差分，可以改用：

- 到达时间与传播距离一致性
- 速度约束
- 相邻点包络时延平滑约束

这条路通常比在不规则点上硬做 `u_xx` 更稳。

## 8. 推荐迁移顺序

### 阶段 1：先改数据结构，不改训练目标

目标：让数据从单点信号升级为多点集合。

要做的事：

1. 新建多接收点数据结构
2. DataLoader 支持可变 `R` 和 mask
3. 保留最基本的监督去噪损失

### 阶段 2：先上 DeepSets 基线

目标：用最小复杂度跑通集合输入。

要做的事：

1. 为每个接收点建立 signal encoder
2. 加入坐标编码器
3. 做 mean/sum pooling
4. 输出逐点去噪结果

交付结果：

- 第一版可训练多接收点模型
- 验证可变 `R` 是否真正跑通

### 阶段 3：在 DeepSets 上加入物理约束

目标：把当前 PINN-style loss 升级为更适合多点场景的损失。

建议顺序：

1. 先保留时间域残差
2. 再尝试空间一致性损失
3. 最后再尝试真正的波动方程近似残差

### 阶段 4：升级到 Set Transformer

目标：利用多点之间的相关性进一步提升性能。

要做的事：

1. 把 pooling 前的独立点表示换成注意力交互表示
2. 增加 receiver mask 支持
3. 对比 DeepSets 与 Set Transformer 的去噪指标和稳定性

## 9. 推荐的实现路线图

```text
Step 1  数据结构升级
single signal -> receiver set

Step 2  DataLoader 升级
fixed tensor -> padded set + mask

Step 3  DeepSets 基线
point encoder + symmetric pooling + point decoder

Step 4  保留当前监督损失
MSE / PSNR / SNR evaluation

Step 5  叠加物理损失
time residual -> spatial consistency -> PDE-like residual

Step 6  升级 Set Transformer
attention over receivers

Step 7  做消融实验
Conv1d vs DeepSets vs Set Transformer
```

## 10. 两条迁移路线怎么选

### 如果你优先考虑低风险迁移

选 DeepSets。

理由：

- 改造量更小
- 更容易调试
- 更适合作为第一版集合基线

### 如果你优先考虑表达能力和多点关系利用

选 Set Transformer。

理由：

- 更擅长处理点间依赖
- 更容易利用传播结构
- 后续更适合复杂物理关系建模

### 实际建议

建议不是二选一，而是顺序推进：

1. 先做 DeepSets 跑通数据和训练流程
2. 再做 Set Transformer 作为增强版本

## 11. 一句话总结

从当前 Conv1d 自编码器迁移出去，真正要改的不是“把 Conv 换成别的层”这么简单，而是把输入建模从“固定单信号张量”升级为“带坐标的接收点集合”。

在这个前提下：

- DeepSets 更适合作为低风险、可变接收点数量的第一版方案
- Set Transformer 更适合作为利用点间关系的增强版方案