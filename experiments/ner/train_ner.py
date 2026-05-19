"""Train a NER model using token classification.

NER (Named Entity Recognition) is different from sentiment classification:
- Classification: whole sentence → 1 label (positive/negative)
- NER: each token → 1 label (B-PER, I-ORG, O, etc.)

Architecture change:
  SequenceClassification head: 768 → 2 (sentiment)
  TokenClassification head:    768 → N (one per NER tag)

The key complexity in NER is label alignment: the tokenizer may split a word
into multiple subword tokens, but NER labels are per-word. We use -100 to
mask subword tokens so the loss ignores them.

Reference: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
  Section 4.2 describes fine-tuning for token-level tasks.

Usage:
  python experiments/ner/train_ner.py
"""

import argparse

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a NER model")
    parser.add_argument("--config", default="configs/training/ner.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # ---- Load data ----
    ds_name = data_cfg["dataset_name"]
    logger.info(f"Loading dataset data/raw/{ds_name} ...")
    ds = load_from_disk(f"data/raw/{ds_name}")

    # Build label list from training data (works for both ClassLabel & string labels)
    raw_labels = set()
    for tags in ds["train"]["ner_tags"]:
        raw_labels.update(tags)
    label_list = sorted(raw_labels)
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    logger.info(f"Labels ({num_labels}): {label_list}")

    # ---- Load model ----
    logger.info(f"Loading model {model_cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ---- Tokenize and align labels ----
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True,
            padding=False, max_length=128,
        )
        labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            tag_ids = [label2id[t] for t in ner_tags]
            aligned = []
            prev_word = None
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != prev_word:
                    aligned.append(tag_ids[word_id])
                else:
                    aligned.append(-100)
                prev_word = word_id
            labels.append(aligned)
        tokenized["labels"] = labels
        return tokenized

    logger.info("Tokenizing and aligning labels ...")
    remove_cols = ["tokens", "pos_tags", "ner_tags"]
    tokenized_ds = ds.map(
        tokenize_and_align_labels, batched=True,
        remove_columns=[c for c in remove_cols if c in ds["train"].column_names],
    )

    # ---- Train ----
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds = [
            [label_list[p] for (p, l) in zip(pred, gold) if l != -100]
            for pred, gold in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (_, l) in zip(pred, gold) if l != -100]
            for pred, gold in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "outputs/checkpoints/ner"),
        num_train_epochs=train_cfg.get("epochs", 5),
        per_device_train_batch_size=train_cfg.get("batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("batch_size", 16),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get(data_cfg.get("eval_split", "validation")),
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting NER training ...")
    trainer.train()
    metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {metrics}")

    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    logger.info(f"Model saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
