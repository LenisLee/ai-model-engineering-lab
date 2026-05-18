"""Launch a training run driven by a YAML config file."""

import argparse

from ai_model_engineering_lab.pipelines.training import TrainingPipeline
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", default="configs/base/config.yaml", help="Path to config file")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Starting training with config: {args.config}")

    pipeline = TrainingPipeline(args.config)
    metrics = pipeline.run()

    logger.info(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
