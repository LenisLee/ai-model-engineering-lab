import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


def main():
    with open("configs/base/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Load data
    ds = load_from_disk(f"data/raw/{data_cfg['dataset_name']}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    metric = evaluate.load("accuracy")

    def tokenize_fn(examples):
        return tokenizer(
            examples[data_cfg["text_column"]], truncation=True, padding="max_length", max_length=512
        )

    ds = ds.map(tokenize_fn, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    num_labels = ds["train"].features[data_cfg["label_column"]].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"], num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"Model saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
