"""Generic HuggingFace Transformers fine-tuning script with full config control."""

import argparse

from ai_model_engineering_lab.pipelines.training import TrainingPipeline
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a HuggingFace model")
    parser.add_argument("--config", default="configs/base/config.yaml", help="Path to config")
    parser.add_argument("--base-model", default=None, help="Override base model")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    logger = setup_logging()

    pipeline = TrainingPipeline(args.config)

    # Apply overrides
    if args.base_model:
        pipeline.model_cfg["base_model"] = args.base_model
    if args.epochs:
        pipeline.train_cfg["epochs"] = args.epochs
    if args.batch_size:
        pipeline.train_cfg["batch_size"] = args.batch_size
    if args.lr:
        pipeline.train_cfg["learning_rate"] = args.lr

    logger.info(f"Fine-tuning {pipeline.model_cfg['base_model']} ...")
    metrics = pipeline.run()
    logger.info(f"Done. Metrics: {metrics}")


if __name__ == "__main__":
    main()
