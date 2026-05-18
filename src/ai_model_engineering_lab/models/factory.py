from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    PreTrainedModel,
)


MODEL_ARCHITECTURES: Dict[str, type] = {
    "sequence_classification": AutoModelForSequenceClassification,
    "causal_lm": AutoModelForCausalLM,
}


def create_model(
    base_model: str,
    num_labels: Optional[int] = None,
    architecture: str = "sequence_classification",
    **kwargs: Any,
) -> PreTrainedModel:
    model_cls = MODEL_ARCHITECTURES.get(architecture)
    if model_cls is None:
        raise ValueError(
            f"Unknown architecture {architecture!r}. Available: {list(MODEL_ARCHITECTURES)}"
        )

    init_kwargs: Dict[str, Any] = dict(kwargs)
    if num_labels is not None:
        init_kwargs["num_labels"] = num_labels

    return model_cls.from_pretrained(base_model, **init_kwargs)


def list_available_models() -> list[str]:
    return list(MODEL_ARCHITECTURES.keys())
