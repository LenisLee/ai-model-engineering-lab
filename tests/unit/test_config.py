import tempfile
from pathlib import Path

import pytest

from ai_model_engineering_lab.utils.config import load_config


class TestLoadConfig:
    def test_load_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnested:\n  item: 42\n")
            path = f.name

        try:
            cfg = load_config(path)
            assert cfg["key"] == "value"
            assert cfg["nested"]["item"] == 42
        finally:
            Path(path).unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
