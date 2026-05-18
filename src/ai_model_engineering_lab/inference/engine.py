from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedModel

from ai_model_engineering_lab.utils.device import get_device


class InferenceEngine:
    """Model inference with batching and device management."""

    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: Any) -> "InferenceEngine":
        from transformers import AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        return cls(model, tokenizer)

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for j, text in enumerate(batch):
                results.append({
                    "text": text,
                    "label": preds[j].item(),
                    "probabilities": probs[j].tolist(),
                })

        return results

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict[str, Any]:
        return self.predict([text])[0]
