## Transformer 内部机制

### 一句话总结

Transformer 的核心操作是：让每个词"看"整句话的所有词，计算哪些词与它最相关，然后把所有词的语义按相关度加权聚合。

### 数据流

```
输入: "这家酒店很好"
         ↓
   Word Embedding (每个字 → 768维向量)
         ↓
   ┌─ Layer 1 ───────────────────────┐
   │  Self-Attention: "很" 在关注谁？   │
   │  ↓                               │
   │  Feed-Forward: 逐位置的线性+激活    │
   └──────────────────────────────────┘
         ↓
   ┌─ Layer 2 ~ Layer 12 ────────────┐
   │  同上，每层越来越抽象               │
   └──────────────────────────────────┘
         ↓
   Pooler: 取 [CLS] token 的输出
         ↓
   Classifier: 768维 → 2维 (正/负概率)
```

### Self-Attention 三步

```
Step 1: 每个词生成 Q/K/V 三个向量
         "酒" → Q_酒, K_酒, V_酒
         "店" → Q_店, K_店, V_店

Step 2: 计算注意力分数
         score(酒→店) = Q_酒 · K_店  / √d_k
         
Step 3: Softmax 归一化 → 加权聚合
         output_酒 = Σ softmax(scores) · V
```

本质：**Q（Query，我要查什么）× K（Key，我有什么）→ 匹配度 → 用 V（Value，实际内容）加权**

### 多头注意力

```
不是一组 Q/K/V，而是 12 组并行的 Q/K/V

Head 1: 可能学到"形容词-名词"关系
Head 2: 可能学到"否定词-被否定词"关系
Head 3: 可能学到"标点-情绪词"关系
...
Head 12: ...
```

每头关注不同的语言现象，最后拼接起来。

### [CLS] Token 的作用

```
[CLS] 是一个特殊的占位 token，不放文字信息

经过 12 层 Transformer 后:
  [CLS] 的 hidden state ≈ 整句话的"语义压缩表示"

分类头直接用 [CLS] 的 768 维向量做 2 分类
```

### 为什么要 12 层

```
Layer 1~3:  学到 surface features (字面组合)
            如: "不"+"好" → 否定短语

Layer 4~6:  学到 syntactic features (句法结构)
            如: 主谓宾结构

Layer 7~9:  学到 semantic features (语义信息)
            如: "酒店环境很好" = 正面情绪

Layer 10~12: task-specific features (任务相关)
            如: 对分类最重要的词
```

### 关键数字

| 概念 | BERT-base Chinese | DistilBERT |
|------|------------------|------------|
| 层数 | 12 | 6 |
| 隐藏维度 | 768 | 768 |
| 注意力头数 | 12 | 12 |
| 每头维度 | 64 | 64 |
| 参数量 | 102M | 67M |
| 最大序列长度 | 512 | 512 |

### 参考论文

1. Vaswani et al. (2017) — *Attention Is All You Need*
   - 提出 Transformer 架构和 Self-Attention 机制
   - Section 3.2 详细描述了 Scaled Dot-Product Attention

2. Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
   - 预训练目标: Masked LM + Next Sentence Prediction
   - Fine-tuning 范式: 预训练底座 + 任务头

3. Clark et al. (2019) — *What Does BERT Look At? An Analysis of BERT's Attention*
   - BERT 的注意力模式分析
   - 发现特定 head 关注特定语言现象（分隔符、依存关系等）

4. Hewitt & Manning (2019) — *A Structural Probe for Finding Syntax in Word Representations*
   - 用线性探针检测隐藏表示中的句法信息
   - 证明 BERT 的中间层编码了完整的句法树
