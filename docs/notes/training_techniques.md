## 训练技巧详解

### 学习率调度 (LR Scheduling)

默认：常量 LR，训练全程用同一个学习率。

```
更好的策略（cosine decay + warmup）：

  LR
   ^
   |     /\
   |    /  \
   |   /    \_________
   |  /               \
   | /                 \
   +---------------------> steps
     warmup   cosine decay
```

**为什么要 warmup**：
训练初期，模型权重是随机的（分类头），梯度很大。如果直接用大 LR，
模型会"爆炸"（loss = NaN）。warmup 用很小的初始 LR 逐步增大，
让优化器先找到合理方向。

**为什么要 decay**：
训练后期接近最优点，需要更小的步长做精细调整。Cosine decay
让 LR 平滑降至接近 0，帮助模型收敛。

配置方式：
```yaml
training:
  lr_scheduler_type: cosine        # linear / cosine / constant
  warmup_steps: 100                # 前100步逐步增加LR
  learning_rate: 0.00002           # 峰值LR
```

### 早停 (Early Stopping)

问题：训练太多 epoch，模型开始"背答案"（过拟合）。
现象：train loss 持续下降，但 eval loss 从某点开始上升。

```
          eval loss
            /\
train loss /  \______
          /
  ← 最佳点    ← 过拟合了
```

早停自动在 eval 不再改善时停止：
```yaml
training:
  early_stopping_patience: 3   # 连续3个eval周期不改善就停
```

### 梯度累积 (Gradient Accumulation)

问题：想用 batch_size=64 但显存只够 batch_size=8。

解决：跑 8 个 batch_size=8，累积梯度不更新，第 8 次再优化器更新。效果等同于 batch_size=64。

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 8   # 有效 batch_size = 8 × 8 = 64
```

### 混合精度 (Mixed Precision)

大部分计算用 fp16（半精度），关键部分保留 fp32。
训练速度翻倍，显存减半，精度几乎无损。

```
fp32: 每个参数 4 bytes → 408MB model
fp16: 每个参数 2 bytes → 204MB model (训练中还要 ×4 for optimizer states)
bf16: 同上但数值范围更大（推荐）
```

```yaml
training:
  bf16: true    # MPS/CPU 上自动禁用，CUDA 上自动开启
```

### 梯度裁剪 (Gradient Clipping)

防止梯度爆炸：如果梯度的 L2 范数超过阈值，按比例缩放。

```yaml
training:
  max_grad_norm: 1.0
```

### 参考

- LR Scheduling: Vaswani et al. (2017) Attention Is All You Need, Section 5.3
- Mixed Precision: Micikevicius et al. (2018) Mixed Precision Training
- Gradient Clipping: Pascanu et al. (2013) On the Difficulty of Training RNNs
