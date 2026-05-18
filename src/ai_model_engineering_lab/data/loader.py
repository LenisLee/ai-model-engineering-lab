from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from datasets import DatasetDict, load_dataset as hf_load, load_from_disk


DATASET_LOADERS: Dict[str, Any] = {}


def register_loader(name: str):
    def decorator(fn):
        DATASET_LOADERS[name] = fn
        return fn

    return decorator


def load_dataset(config: Dict[str, Any]) -> DatasetDict:
    name = config.get("dataset_name", "")
    split = config.get("split")

    loader = DATASET_LOADERS.get(name)
    if loader:
        return loader(config)

    raw_path = Path(f"data/raw/{name}")
    if raw_path.exists():
        ds = load_from_disk(str(raw_path))
    else:
        ds = hf_load(name, split=split)

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds} if split else {"data": ds})
    return ds
