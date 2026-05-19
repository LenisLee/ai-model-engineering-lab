## 学习路线图

按从浅到深的顺序，每一层都配有可运行的代码和参考论文。

---

### Phase 1: Foundation ✅

- [x] Project scaffolding & directory structure
- [x] Config-driven IMDB training pipeline (English)
- [x] Chinese sentiment analysis with bert-base-chinese (92.3% accuracy)
- [x] Core interfaces & protocols
- [x] Registry pattern, data loaders, model factory
- [x] Training, inference, evaluation modules
- [x] End-to-end pipeline abstractions

### Phase 2: Understanding the Black Box 🚧

- [ ] **Attention Visualization** — 把 Transformer 每一层的注意力权重画成热力图，看模型在"关注"哪些词
  - 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
  - 论文: [What Does BERT Look At?](https://arxiv.org/abs/1906.04341) (Clark et al., 2019)
  - 实验: `experiments/attention_viz/` — 用 heatmap 可视化 cls token 的 attention
- [ ] **Probe Classifier** — 取中间层 hidden states 训练线性分类器，看语义信息在哪一层形成
  - 论文: [A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/) (Hewitt & Manning, 2019)
  - 实验: `experiments/attention_viz/` — 线性探针检测情感信息

### Phase 3: New Task Types ✅

- [ ] **NER (命名实体识别)** — 从"一句话一个标签"到"每个 token 一个标签"，学习 token classification
  - 数据集: peoples_daily_ner (人民日报中文 NER)
  - 论文: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2019)
  - 实验: `experiments/ner/` — 完整的 NER 训练+评估
- [x] **Baselines** — TF-IDF + Logistic Regression lower-bound
- [x] **HF Transformers** — Config-driven fine-tuning with model overrides
- [x] **LoRA** — PEFT-based parameter-efficient fine-tuning
  - 论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

### Phase 4: Training Engineering 🚧

- [ ] **学习率调度** — 从常量 LR 到 warmup + cosine decay，理解为什么 LR 调度重要
  - 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Section 5.3 (warmup formula)
- [ ] **早停 (Early Stopping)** — 自动在过拟合前停止训练，保存最优 checkpoint
- [ ] **梯度累积** — 用小的 per-device batch 模拟大 batch 训练
- [ ] **混合精度 (fp16/bf16)** — 训练加速 2×，显存减半
  - 论文: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2018)

### Phase 5: Experiment Tracking 🚧

- [ ] **MLflow 集成** — 每次训练自动记录：参数、metrics、模型 artifact
  - 可对比多次实验的 loss 曲线和准确率
- [ ] **实验结果对比** — 同一模型不同超参的横向对比

### Phase 6: RAG & Agents ✅

- [x] **RAG** — FAISS 向量索引 + SentenceTransformer 检索
  - 论文: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [x] **Agents** — LLM tool-calling agent loop
  - 论文: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)

### Phase 7: Production 🚧

- [ ] **模型量化** — float32 → int8 / int4，模型缩小 4-8 倍，推理加速 2-4 倍
  - 工具: bitsandbytes / llama.cpp
  - 论文: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [ ] **ONNX 导出** — 模型转为跨平台标准格式，脱离 PyTorch 运行时
- [ ] **Docker 部署** — FastAPI + Gradio 容器化，`docker run` 一键启动
- [ ] **Gradio** — Interactive text classification demo
- [ ] **FastAPI** — Inference REST API with `/predict` endpoint

---

## 论文清单

| # | 论文 | 关键概念 | 年份 |
|---|------|---------|------|
| 1 | Attention Is All You Need | Transformer, Self-Attention, Warmup | 2017 |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers | MLM, NSP, Fine-tuning | 2019 |
| 3 | What Does BERT Look At? | Attention 模式分析, 语言学探针 | 2019 |
| 4 | A Structural Probe for Finding Syntax | 线性探针, 表征分析 | 2019 |
| 5 | Mixed Precision Training | fp16, loss scaling | 2018 |
| 6 | LoRA: Low-Rank Adaptation | 参数高效微调, 低秩分解 | 2021 |
| 7 | RAG for Knowledge-Intensive NLP Tasks | 检索增强生成 | 2020 |
| 8 | ReAct: Synergizing Reasoning and Acting | Tool use, Agent 推理 | 2022 |
| 9 | QLoRA: Efficient Finetuning of Quantized LLMs | 量化微调, NF4 | 2023 |
