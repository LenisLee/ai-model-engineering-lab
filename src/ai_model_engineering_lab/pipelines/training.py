from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from datasets import load_from_disk

from ai_model_engineering_lab.data.loader import load_dataset
from ai_model_engineering_lab.data.preprocessing import TextPreprocessor
from ai_model_engineering_lab.evaluation.metrics import make_compute_metrics_fn
from ai_model_engineering_lab.models.factory import create_model
from ai_model_engineering_lab.training.trainer import HFTrainer
from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging


class TrainingPipeline:
    """End-to-end training pipeline driven by a config file."""

    def __init__(self, config_path: str | Path):
        self.config = load_config(config_path)
        self.logger = setup_logging()

        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.train_cfg = self.config.get("training", {})

    def run(self) -> Dict[str, Any]:
        self.logger.info("Loading dataset ...")
        ds = self._load_data()

        eval_split = self.data_cfg.get("eval_split", "test")
        max_train = self.data_cfg.get("max_train_samples")
        max_eval = self.data_cfg.get("max_eval_samples")
        if max_train:
            ds["train"] = ds["train"].shuffle(seed=42).select(range(min(max_train, len(ds["train"]))))
        if max_eval and eval_split in ds:
            ds[eval_split] = ds[eval_split].shuffle(seed=42).select(range(min(max_eval, len(ds[eval_split]))))

        for split_name, split_ds in ds.items():
            self.logger.info(f"  {split_name}: {len(split_ds)} samples")

        num_labels = self._infer_num_labels(ds)

        self.logger.info(f"Creating model '{self.model_cfg.get('base_model')}' with {num_labels} labels ...")
        model = create_model(
            base_model=self.model_cfg["base_model"],
            num_labels=num_labels,
            architecture=self.model_cfg.get("architecture", "sequence_classification"),
        )

        self.logger.info("Tokenizing dataset ...")
        tokenized_ds = self._preprocess(ds)

        self.logger.info("Starting training ...")
        trainer = HFTrainer.from_config(
            config=self.config,
            model=model,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds.get(eval_split),
            compute_metrics=make_compute_metrics_fn(),
        )

        # Optional MLflow tracking
        tracker = None
        if self.train_cfg.get("mlflow_tracking", False):
            from ai_model_engineering_lab.training.tracker import ExperimentTracker
            tracker = ExperimentTracker()
            if tracker.start_run(run_name=f"{self.model_cfg.get('base_model')}_{self.data_cfg.get('dataset_name')}"):
                tracker.log_params(self.config)
                self.logger.info("MLflow tracking enabled")

        train_metrics = trainer.train()
        eval_metrics = trainer.evaluate()

        if tracker:
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    tracker.log_metrics({k: v})
            tracker.end_run()

        trainer.save()
        self.logger.info(f"Model saved to {self.train_cfg.get('output_dir', 'outputs/checkpoints')}")

        return {"train": train_metrics, "eval": eval_metrics}

    def _load_data(self):
        dataset_name = self.data_cfg.get("dataset_name", "")
        raw_path = Path(f"data/raw/{dataset_name}")
        if raw_path.exists():
            return load_from_disk(str(raw_path))
        return load_dataset(self.data_cfg)

    def _preprocess(self, ds):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_cfg["base_model"])
        preprocessor = TextPreprocessor(
            tokenizer=tokenizer,
            max_length=self.data_cfg.get("max_length", 512),
            text_column=self.data_cfg.get("text_column", "text"),
            label_column=self.data_cfg.get("label_column", "label"),
        )
        return ds.map(preprocessor, batched=True)

    @staticmethod
    def _infer_num_labels(ds) -> int:
        if "train" in ds and hasattr(ds["train"].features.get("label"), "num_classes"):
            return ds["train"].features["label"].num_classes
        return 2  # default binary
