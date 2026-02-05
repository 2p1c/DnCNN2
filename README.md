# Hybrid Polyglot AI Template

一个为全栈智能开发设计的混合语言项目模板，集成了前端敏捷开发（TypeScript/React）、后端算力（PyTorch/ML）和 LLM API 集成的最佳实践。

## 🎯 特性

- **前端开发标准**: React/Next.js + TypeScript + Tailwind CSS 最佳实践
- **深度学习工程**: PyTorch 神经网络开发规范，强制 Tensor 形状注释
- **LLM 集成**: OpenAI/Anthropic API 调用、流式传输和 Prompt 工程
- **自动化工作流**: Git 自动提交脚本，符合 Conventional Commits 规范
- **认知协议**: 代码质量控制和可视化思维指南

## 📂 项目结构

```
.
├── .agent/
│   ├── rules/          # Agent 行为规则和编码标准
│   │   ├── 00_meta_behavior.md
│   │   ├── 10_frontend_web.md
│   │   ├── 20_ml_pytorch.md
│   │   ├── 30_llm_api.md
│   │   └── 90_git_workflow.md
│   └── skills/         # 自定义技能和工具脚本
├── scripts/
│   └── git_auto.py     # Git 自动化脚本
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 前端项目

```bash
# 使用 Vite 创建 React + TypeScript 项目
npx -y create-vite@latest ./ --template react-ts

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### Python/ML 项目

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 安装依赖
pip install torch numpy pandas
```

## 🛠️ Git 自动提交

使用内置脚本快速提交代码：

```bash
python scripts/git_auto.py "feat: 添加新功能"
python scripts/git_auto.py "fix: 修复登录问题"
```

## 📖 编码规范

所有规则文档位于 `.agent/rules/` 目录：

- **00_meta_behavior.md**: 核心认知协议，强调理解优先和可视化思维
- **10_frontend_web.md**: React/TypeScript/Tailwind CSS 开发标准
- **20_ml_pytorch.md**: PyTorch 最佳实践，包含 Tensor 形状注释要求
- **30_llm_api.md**: LLM API 集成和 Prompt 工程指南
- **90_git_workflow.md**: Git 自动化工作流程

## 💡 使用建议

1. **作为 GitHub 模板**: 在 GitHub Settings 中勾选 "Template repository"
2. **与 AI Agent 协作**: 配合 Antigravity 等 AI 编码助手使用，Agent 会自动读取规则
3. **渐进式采用**: 根据项目需求选择性使用规则文件

## 📄 许可证

MIT License
