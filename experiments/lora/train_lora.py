"""LoRA fine-tuning with PEFT on a sequence classification task."""

import argparse

from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

from ai_model_engineering_lab.data.preprocessing import TextPreprocessor
from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base/config.yaml")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Load data
    ds = load_from_disk(f"data/raw/{data_cfg['dataset_name']}")
    num_labels = ds["train"].features[data_cfg["label_column"]].num_classes

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    preprocessor = TextPreprocessor(
        tokenizer=tokenizer, text_column=data_cfg["text_column"], label_column=data_cfg["label_column"]
    )
    ds = ds.map(preprocessor, batched=True)

    # Model with LoRA
    logger.info(f"Loading base model {model_cfg['base_model']} with LoRA r={args.lora_r} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"], num_labels=num_labels
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            **accuracy.compute(predictions=preds, references=labels),
            **f1.compute(predictions=preds, references=labels, average="macro"),
        }

    # Train
    args_training = TrainingArguments(
        output_dir="outputs/checkpoints/lora",
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args_training,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting LoRA training ...")
    trainer.train()
    trainer.save_model("outputs/checkpoints/lora")
    logger.info("LoRA model saved to outputs/checkpoints/lora")


if __name__ == "__main__":
    main()
