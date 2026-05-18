"""Baseline: train a basic classifier as a lower-bound reference."""

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from datasets import load_from_disk
from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base/config.yaml")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    logger.info(f"Loading {data_cfg['dataset_name']} ...")
    ds = load_from_disk(f"data/raw/{data_cfg['dataset_name']}")

    train_texts = ds["train"][data_cfg["text_column"]]
    train_labels = ds["train"][data_cfg["label_column"]]
    test_texts = ds["test"][data_cfg["text_column"]]
    test_labels = ds["test"][data_cfg["label_column"]]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    logger.info("Training baseline model ...")
    model.fit(train_texts, train_labels)

    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")

    logger.info(f"Baseline results — Accuracy: {acc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
