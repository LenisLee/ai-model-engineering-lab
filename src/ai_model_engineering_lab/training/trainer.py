from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import SchedulerType


DEFAULT_TRAINING = {
    "output_dir": "outputs/checkpoints",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "logging_steps": 100,
    "num_workers": 0,
    "fp16": False,
    "bf16": False,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "early_stopping_patience": 0,
}


class HFTrainer:
    """Wraps HuggingFace Trainer with full training technique support.

    Supports: LR scheduling, early stopping, gradient accumulation,
              mixed precision (fp16/bf16), warmup, gradient clipping.
    """

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
        train_cfg = {**DEFAULT_TRAINING, **cfg.get("training", {})}

        self.output_dir = train_cfg["output_dir"]

        # Auto-detect best precision
        fp16 = train_cfg["fp16"]
        bf16 = train_cfg["bf16"]
        if not fp16 and not bf16 and torch.cuda.is_available():
            fp16 = torch.cuda.is_bf16_supported()
            if fp16:
                fp16, bf16 = False, True

        callbacks = []
        early_stop = train_cfg.get("early_stopping_patience", 0)
        if early_stop > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stop)
            )

        lr_scheduler = train_cfg.get("lr_scheduler_type", "linear")
        try:
            scheduler_type = SchedulerType(lr_scheduler)
        except ValueError:
            scheduler_type = SchedulerType.LINEAR

        self.args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            per_device_eval_batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            eval_strategy=train_cfg["eval_strategy"],
            save_strategy=train_cfg["save_strategy"],
            load_best_model_at_end=train_cfg["load_best_model_at_end"],
            metric_for_best_model=train_cfg["metric_for_best_model"],
            logging_steps=train_cfg["logging_steps"],
            fp16=fp16,
            bf16=bf16,
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            warmup_steps=train_cfg.get("warmup_steps", 0),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
            lr_scheduler_type=scheduler_type,
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            dataloader_num_workers=train_cfg["num_workers"],
            report_to="none",
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
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
