from __future__ import annotations

from typing import Any, Callable, Dict

import evaluate
import numpy as np


def load_metric(name: str) -> Any:
    return evaluate.load(name)


def compute_classification_metrics(
    predictions: np.ndarray, references: np.ndarray
) -> Dict[str, float]:
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    return {
        **accuracy.compute(predictions=predictions, references=references),
        **f1.compute(predictions=predictions, references=references, average="macro"),
    }


def make_compute_metrics_fn(tokenizer: Any = None) -> Callable:
    """Factory for the standard HuggingFace compute_metrics callback."""

    def compute_metrics_fn(eval_pred: tuple) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return compute_classification_metrics(predictions, labels)

    return compute_metrics_fn


compute_metrics = make_compute_metrics_fn()
