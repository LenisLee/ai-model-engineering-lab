"""Download a dataset and save it to data/raw/<dataset_name>."""

import argparse

from datasets import load_dataset

from ai_model_engineering_lab.utils.config import load_config
from ai_model_engineering_lab.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download a dataset")
    parser.add_argument("--config", default="configs/base/config.yaml", help="Path to config file")
    parser.add_argument("--dataset", default=None, help="Override dataset name from config")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)
    ds_name = args.dataset or cfg["data"]["dataset_name"]

    logger.info(f"Downloading {ds_name} ...")
    ds = load_dataset(ds_name)
    ds.save_to_disk(f"data/raw/{ds_name}")
    logger.info(f"Saved to data/raw/{ds_name}")


if __name__ == "__main__":
    main()
