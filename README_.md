# 🛢️ Oil Stain Detection System (YOLOv8 + MLOps)

本项目用于检测石子表面的油渍，基于 YOLOv8 构建完整的工业级检测流程，包含：

- 📊 数据标注（Roboflow）
- 🏋️ 模型训练（YOLOv8）
- 🔍 推理服务（FastAPI）
- 📦 容器化部署（Docker）
- ☁️ 云效流水线执行（CI/CD）
- 🗂️ 制品仓库存储数据与模型

---

# 📦 项目结构

```text
Oil_stain_Detection/
├── oil_stain_detection/   # 核心代码
│   ├── app.py             # API服务
│   ├── train.py           # 训练模块
│   ├── run_inference.py   # 推理模块
│   ├── model_tasks.json   # 模型配置
│   ├── weights/           # 模型权重
│   ├── data/              # 数据集
│   ├── input/             # 输入数据
│   └── outputs/           # 输出结果
├── Dockerfile             # 容器部署
└── README.md              # 项目说明
```

---

# 📊 数据集

- 标注工具：Roboflow
- 格式：YOLOv8
- 存储方式：云效制品仓库（zip）

---

# 🏋️ 模型训练

## 1️⃣ 安装依赖
pip install ultralytics
## 2️⃣ 启动训练
cd oil_stain_detection
python train.py
## 3️⃣ 输出结果
训练完成后生成：

runs/train/oil_stain/
└── weights/
    ├── best.pt
    └── last.pt

---

# 🔍 推理（离线批处理）

适用于视频抽帧分析或图片批量检测：
cd oil_stain_detection
python run_inference.py

输出：

outputs/<timestamp>/
├── result.json
└── 可视化检测图

---

# 🌐 推理服务（API）

## 启动服务
cd oil_stain_detection
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

## API接口

  健康检查
  curl http://127.0.0.1:8000/health
  
  单图检测
  curl -X POST "http://127.0.0.1:8000/predict" \
    -F "file=@test.jpg"
  
  批量检测
  curl -X POST "http://127.0.0.1:8000/predict/batch" \
    -F "files=@a.jpg" \
    -F "files=@b.jpg"

---

# ☁️ 云效部署流程

## 1️⃣ 构建镜像
docker build -t oil-stain-infer:1.0 .
## 2️⃣ 推送到 ACR
docker tag oil-stain-infer:1.0 crpi-jyr4o5mitu1zgg78.cn-beijing.personal.cr.aliyuncs.com/yang_xudong/oil-stain-infer:1.0

docker push crpi-jyr4o5mitu1zgg78.cn-beijing.personal.cr.aliyuncs.com/yang_xudong/oil-stain-infer:1.0
## 3️⃣ 云效流水线执行

镜像地址：
crpi-jyr4o5mitu1zgg78.cn-beijing.personal.cr.aliyuncs.com/yang_xudong/oil-stain-infer:1.0

执行命令：
cd oil_stain_detection
python run_inference.py

---

# 📂 数据接入（制品仓库）

在云效流水线中：
unzip oil-stain-train-v1.zip -d data

---

# ⚠️ 注意事项

Python >= 3.8

推荐 GPU 训练

云效适合推理，不适合长时间训练

权重文件需放在 weights/ 目录

---

# 🚀 技术亮点

基于 YOLOv8 实现油渍检测模型

使用 FastAPI 构建推理服务（/predict /batch）

Docker 容器化部署 + ACR 镜像仓库

云效流水线实现自动推理

JSON结构化输出，支持系统集成

支持批量图像处理与结果持久化

---

# 🔮 后续优化方向

🎯 提升检测精度（数据增强 + 调参）

🎥 视频自动抽帧检测

📊 油渍面积与分布统计

🤖 对接机器人路径规划

⚡ GPU 推理加速

---

# 👨‍💻 Author

Yang Xudong

### 3 分钟了解如何进入开发

欢迎使用云效代码管理 Codeup，通过阅读以下内容，你可以快速熟悉 Codeup ，并立即开始今天的工作。

### 提交**文件**

Codeup 支持两种方式进行代码提交：网页端提交，以及本地 Git 客户端提交。

* 如需体验本地命令行操作，请先安装 Git 工具，安装方法参见[安装Git](https://help.aliyun.com/document_detail/153800.html)。

* 如需体验 SSH 方式克隆和提交代码，请先在平台账号内配置 SSH 公钥，配置方法参见[配置 SSH 密钥](https://help.aliyun.com/document_detail/153709.html)。

* 如需体验 HTTP 方式克隆和提交代码，请先在平台账号内配置克隆账密，配置方法参见[配置 HTTPS 克隆账号密码](https://help.aliyun.com/document_detail/153710.html)。

现在，你可以在 Codeup 中提交代码文件了，跟着文档「[__提交第一行代码__](https://help.aliyun.com/document_detail/153707.html?spm=a2c4g.153710.0.0.3c213774PFSMIV#6a5dbb1063ai5)」一起操作试试看吧。

<img src="https://img.alicdn.com/imgextra/i3/O1CN013zHrNR1oXgGu8ccvY_!!6000000005235-0-tps-2866-1268.jpg" width="100%" />


### 进行代码检测

开发过程中，为了更好的维护你的代码质量，你可以开启 Codeup 内置开箱即用的「[代码检测服务](https://help.aliyun.com/document_detail/434321.html)」，开启后提交或合并请求的变更将自动触发检测，识别代码编写规范和安全漏洞问题，并及时提供结果报表和修复建议。

<img src="https://img.alicdn.com/imgextra/i2/O1CN01BRzI1I1IO0CR2i4Aw_!!6000000000882-0-tps-2862-1362.jpg" width="100%" />

### 开展代码评审

功能开发完毕后，通常你需要发起「[代码评审并执行合并](https://help.aliyun.com/document_detail/153872.html)」，Codeup 支持多人协作的代码评审服务，你可以通过「[保护分支设置合并规则](https://help.aliyun.com/document_detail/153873.html?spm=a2c4g.203108.0.0.430765d1l9tTRR#p-4on-aep-l5q)」策略及「[__合并请求设置__](https://help.aliyun.com/document_detail/153874.html?spm=a2c4g.153871.0.0.3d38686cJpcdJI)」对合并过程进行流程化管控，同时提供在线代码评审及冲突解决能力，让评审过程更加流畅。

<img src="https://img.alicdn.com/imgextra/i1/O1CN01MaBDFH1WWcGnQqMHy_!!6000000002796-0-tps-2592-1336.jpg" width="100%" />

### 成员协作

是时候邀请成员一起编写卓越的代码工程了，请点击左下角「成员」邀请你的小伙伴开始协作吧！

### 更多

Git 使用教学、高级功能指引等更多说明，参见[Codeup帮助文档](https://help.aliyun.com/document_detail/153402.html)。