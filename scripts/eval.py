"""Evaluate a trained model on the test set."""

import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from ai_model_engineering_lab.data.preprocessing import TextPreprocessor
from ai_model_engineering_lab.evaluation.metrics import make_compute_metrics_fn
from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", default="configs/base/config.yaml", help="Path to config file")
    parser.add_argument("--model", default=None, help="Override model path")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_path = args.model or train_cfg["output_dir"]

    logger.info(f"Loading dataset from data/raw/{data_cfg['dataset_name']} ...")
    ds = load_from_disk(f"data/raw/{data_cfg['dataset_name']}")

    logger.info(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    preprocessor = TextPreprocessor(
        tokenizer=tokenizer,
        max_length=512,
        text_column=data_cfg["text_column"],
        label_column=data_cfg["label_column"],
    )
    ds = ds.map(preprocessor, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_path,
            per_device_eval_batch_size=train_cfg.get("batch_size", 16),
        ),
        compute_metrics=make_compute_metrics_fn(),
    )

    results = trainer.evaluate(eval_dataset=ds["test"])
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()
