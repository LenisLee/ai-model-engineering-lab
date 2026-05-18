from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ai_model_engineering_lab.inference.engine import InferenceEngine
from ai_model_engineering_lab.utils.logging import setup_logging


class InferencePipeline:
    """End-to-end inference pipeline."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.logger = setup_logging()

    def run(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading model from {self.model_path} ...")
        engine = InferenceEngine.from_pretrained(str(self.model_path))
        return engine.predict(texts, batch_size=batch_size)
