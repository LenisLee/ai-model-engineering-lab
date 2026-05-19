"""Linear probe: train a simple classifier on each layer's hidden states.

Reveals where semantic information (sentiment) is encoded in the network.
If a linear classifier on layer N achieves high accuracy, then layer N already
contains linearly separable sentiment information.

Reference:
  "A Structural Probe for Finding Syntax in Word Representations"
  (Hewitt & Manning, 2019)

Usage:
  python experiments/attention_viz/probe.py \
    --model outputs/checkpoints/chinese_sentiment \
    --max-samples 500
"""

import argparse
from typing import Dict, List

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ai_model_engineering_lab.utils.device import get_device


@torch.no_grad()
def extract_hidden_states(
    model, tokenizer, texts: List[str], batch_size: int = 8
) -> Dict[int, np.ndarray]:
    """Extract [CLS] hidden states from each layer for a batch of texts."""
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    layer_states: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_layers + 1)}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: (embedding + each layer), each (batch, seq_len, hidden_dim)
        all_hidden = outputs.hidden_states

        for layer_idx, hidden in enumerate(all_hidden):
            cls_state = hidden[:, 0, :].cpu().numpy()  # [CLS] token
            layer_states[layer_idx].append(cls_state)

    return {
        layer: np.concatenate(states, axis=0)
        for layer, states in layer_states.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Train linear probes per layer")
    parser.add_argument("--model", default="outputs/checkpoints/chinese_sentiment",
                        help="Path to trained model")
    parser.add_argument("--dataset", default="chnsenticorp",
                        help="Dataset name in data/raw/")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples to use (half train, half test)")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, output_hidden_states=True
    ).to(device)
    model.eval()

    print(f"Loading dataset data/raw/{args.dataset} ...")
    ds = load_from_disk(f"data/raw/{args.dataset}")
    test_ds = ds["test"]

    n = min(args.max_samples, len(test_ds))
    texts = test_ds["text"][:n]
    labels = test_ds["label"][:n]

    print(f"Extracting hidden states for {n} samples ({model.config.num_hidden_layers + 1} layers) ...")
    layer_states = extract_hidden_states(model, tokenizer, texts)

    split = n // 2
    X_train = {l: s[:split] for l, s in layer_states.items()}
    X_test = {l: s[split:] for l, s in layer_states.items()}
    y_train, y_test = labels[:split], labels[split:]

    print(f"\nLayer-wise probe accuracy:")
    print(f"{'Layer':<8} {'Accuracy':<12} Description")
    print("-" * 50)

    for layer_idx in range(model.config.num_hidden_layers + 1):
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train[layer_idx], y_train)
        preds = clf.predict(X_test[layer_idx])
        acc = accuracy_score(y_test, preds)

        if layer_idx == 0:
            desc = "Word embeddings (before any Transformer layer)"
        elif layer_idx == model.config.num_hidden_layers:
            desc = "Final hidden state → classifier input"
        else:
            desc = f"After Transformer layer {layer_idx}"

        bar = "█" * int(acc * 40)
        print(f"  {layer_idx:<6} {acc:.4f}      {bar} {desc}")

    print("\nKey insight: accuracy jumps up → that layer learned the task-relevant features.")


if __name__ == "__main__":
    main()
