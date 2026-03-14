# 基于神经网络去噪的非接触式激光超声缺陷检测

> 完整项目代码：[Laser-NDT-Based-NN](https://github.com/2p1c/Laser-NDT-Based-NN.git).

## 项目概述
### 项目背景
非接触式激光超声检测（Laser Ultrasonic Testing, LUT）是一种先进的无损检测技术，利用激光产生和接收超声波信号来检测材料中的缺陷。然而，LUT信号通常受到环境噪声和系统噪声的干扰，导致信噪比（SNR）较低，影响检测的准确性和可靠性。

激光超声依靠热膨胀效应产生超声波，但大量研究表明，实际实验中难以把控激光参数和环境因素，不同材料之间的热膨胀效应和烧蚀阈值。激光器的电压，聚集光斑大小，材料的热扩散特性，激励频率等等因素都会影响生成激光信号的质量和一致性。

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
