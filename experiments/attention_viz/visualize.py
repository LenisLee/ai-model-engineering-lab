"""Visualize Transformer attention weights as heatmaps.

References:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - "What Does BERT Look At?" (Clark et al., 2019)

Usage:
  python experiments/attention_viz/visualize.py \
    --model outputs/checkpoints/chinese_sentiment \
    --text "这家酒店环境很好，服务态度也不错"
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ai_model_engineering_lab.utils.device import get_device


def visualize_attention(
    model,
    tokenizer,
    text: str,
    layer: int = -1,
    head: int = 0,
    output_path: str = "outputs/attention_heatmap.png",
):
    """Extract and visualize attention weights for a given text.

    Shows what tokens the [CLS] token attends to — revealing which words
    the model considers most important for classification.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attention shape: (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions  # list of 6 or 12 layers

    # Visualize each layer's CLS attention
    num_layers = len(attentions)
    fig, axes = plt.subplots(3, (num_layers + 2) // 3, figsize=(16, 10))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        # Get attention from [CLS] to all tokens, average across heads
        cls_attn = attentions[layer_idx][0, :, 0, :].mean(dim=0).cpu().numpy()

        ax.bar(range(len(tokens)), cls_attn, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_ylabel("Attn from [CLS]")

    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Attention from [CLS] token\nText: {text[:80]}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")

    # Print top-5 attended tokens for the final layer
    print(f"\nTop attended tokens in final layer (avg across all heads):")
    cls_attn_final = attentions[-1][0, :, 0, :].mean(dim=0).cpu().numpy()
    top_idx = cls_attn_final.argsort()[::-1][:5]
    for idx in top_idx:
        print(f"  {tokens[idx]:<20} weight={cls_attn_final[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Transformer attention")
    parser.add_argument("--model", default="outputs/checkpoints/chinese_sentiment",
                        help="Path to trained model")
    parser.add_argument("--text", default="这家酒店环境很好，服务态度也不错",
                        help="Input text to analyze")
    parser.add_argument("--output", default="outputs/attention_heatmap.png",
                        help="Output image path")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, attn_implementation="eager"
    ).to(device)
    model.eval()

    visualize_attention(model, tokenizer, args.text, output_path=args.output)


if __name__ == "__main__":
    main()
