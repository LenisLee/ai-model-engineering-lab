# ai-model-engineering-lab

AI 模型工程实验平台，用于模型训练、推理、评估和应用构建的结构化工作区。

## 目录结构

```
├── src/ai_model_engineering_lab/   # Python 包
│   ├── core/                       # 核心抽象与基类
│   ├── models/                     # 模型定义与注册
│   ├── data/                       # 数据加载、预处理、增强
│   ├── training/                   # 训练循环、优化器、调度器
│   ├── inference/                  # 推理引擎与批处理
│   ├── evaluation/                 # 评估指标与基准测试
│   ├── pipelines/                  # 端到端训练/推理流水线
│   ├── utils/                      # 通用工具函数
│   └── interfaces/                 # 接口定义与协议
├── scripts/                        # 命令行入口
│   ├── download_data.py            # 数据集下载脚本
│   ├── train.py                    # 训练启动脚本
│   ├── infer.py                    # 推理脚本
│   └── eval.py                     # 评估脚本
├── configs/                        # 配置文件
│   ├── base/                       # 基础配置模板
│   ├── training/                   # 训练配置
│   ├── inference/                   # 推理配置
│   └── evaluation/                 # 评估配置
├── experiments/                    # 实验代码
│   ├── hf_transformers/            # HuggingFace Transformers 实验
│   ├── lora/                       # LoRA 微调实验
│   ├── rag/                        # RAG 检索增强生成实验
│   ├── agents/                     # AI Agent 实验
│   └── baselines/                  # 基线模型实验
├── notebooks/                      # Jupyter Notebooks
│   ├── 00_exploration/             # 数据探索
│   ├── 01_data_analysis/           # 数据分析
│   └── 02_experiments/             # 实验记录
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── interim/                    # 中间处理数据
│   ├── processed/                  # 处理后数据
│   └── external/                   # 外部数据源
├── outputs/                        # 输出目录
│   ├── checkpoints/                # 模型检查点
│   ├── logs/                       # 训练日志
│   ├── metrics/                    # 评估指标
│   └── predictions/                # 推理结果
├── tests/                          # 测试
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   └── fixtures/                   # 测试数据
├── resources/                      # 资源文件
│   ├── diagrams/                   # 架构图
│   ├── datasets/                   # 数据集卡片
│   └── model_cards/                # 模型卡片
├── docs/                           # 文档
│   ├── experiments/                # 实验文档
│   ├── notes/                      # 技术笔记
│   └── roadmap.md                  # 路线图
└── app/                            # 应用
    ├── gradio_app.py               # Gradio 演示应用
    └── api/                        # API 服务
```

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发模式
pip install -e .
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `python scripts/download_data.py` | 下载数据集 |
| `python scripts/train.py` | 启动训练 |
| `python scripts/infer.py` | 运行推理 |
| `python scripts/eval.py` | 运行评估 |

## 实验方向

- **Transformers 微调** — 使用 HuggingFace Transformers 进行模型微调
- **LoRA** — 低秩适配器进行参数高效微调
- **RAG** — 检索增强生成的实现与优化
- **Agent** — LLM Agent 的实验与评估
- **Baseline** — 各方向基线模型的建立与对比

## 命名规范

- 目录与文件名统一使用 `snake_case`
- 实验子目录按技术主题命名（如 `lora`、`rag`）
- 配置模板使用 YAML 格式
