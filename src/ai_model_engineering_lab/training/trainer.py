from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)


class HFTrainer:
    """Wraps HuggingFace Trainer with configuration-driven setup."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Any,
        eval_dataset: Any = None,
        compute_metrics: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

        cfg = config or {}
        train_cfg = cfg.get("training", {})

        self.output_dir = train_cfg.get("output_dir", "outputs/checkpoints")
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_cfg.get("epochs", 3),
            per_device_train_batch_size=train_cfg.get("batch_size", 8),
            per_device_eval_batch_size=train_cfg.get("batch_size", 8),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
            evaluation_strategy=train_cfg.get("evaluation_strategy", "epoch"),
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
            logging_dir=train_cfg.get("logging_dir", "outputs/logs"),
            logging_steps=train_cfg.get("logging_steps", 100),
            fp16=train_cfg.get("fp16", torch.cuda.is_available()),
            dataloader_num_workers=train_cfg.get("num_workers", 0),
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self) -> Dict[str, float]:
        result = self.trainer.train()
        return result.metrics

    def evaluate(self) -> Dict[str, float]:
        return self.trainer.evaluate()

    def save(self) -> None:
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        model: PreTrainedModel,
        train_dataset: Any,
        eval_dataset: Any = None,
        compute_metrics: Optional[Callable] = None,
    ) -> "HFTrainer":
        model_cfg = config.get("model", {})
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
        return cls(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            config=config,
        )
