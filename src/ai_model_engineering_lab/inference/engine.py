from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedModel

from ai_model_engineering_lab.utils.device import get_device


class InferenceEngine:
    """Model inference with batching, device management, and quantization.

    Supports:
      - Standard float32 inference
      - bitsandbytes 8-bit quantization (load_in_8bit=True)
      - bitsandbytes 4-bit quantization (load_in_4bit=True)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> "InferenceEngine":
        """Load a model with optional quantization.

        Args:
            model_path: Path to saved model directory.
            load_in_8bit: Use 8-bit quantization (4x memory reduction).
            load_in_4bit: Use 4-bit quantization (8x memory reduction).

        Quantization reference: "QLoRA: Efficient Finetuning of Quantized LLMs"
        (Dettmers et al., 2023)
        """
        from transformers import AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs: Dict[str, Any] = dict(kwargs)

        if load_in_8bit or load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install with: pip install bitsandbytes"
                )

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, **model_kwargs
        )
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

    def export_onnx(self, output_path: str, max_length: int = 512) -> None:
        """Export model to ONNX format for cross-platform deployment.

        ONNX enables running the model without PyTorch, e.g. with ONNX Runtime.
        """
        dummy_input = self.tokenizer(
            "Hello world", truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        input_tuple = (
            dummy_input["input_ids"].to(self.device),
            dummy_input["attention_mask"].to(self.device),
        )

        torch.onnx.export(
            self.model,
            input_tuple,
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
        )
