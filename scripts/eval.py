import yaml
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate


def main():
    with open("configs/base/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    ds = load_from_disk(f"data/raw/{data_cfg['dataset_name']}")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["output_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(train_cfg["output_dir"])

    def tokenize_fn(examples):
        return tokenizer(
            examples[data_cfg["text_column"]], truncation=True, padding="max_length", max_length=512
        )

    ds = ds.map(tokenize_fn, batched=True)

    metric = evaluate.combine(["accuracy", "f1"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=train_cfg["output_dir"], per_device_eval_batch_size=16),
    )

    results = trainer.evaluate(eval_dataset=ds["test"])
    print("Eval results:", results)


if __name__ == "__main__":
    main()
