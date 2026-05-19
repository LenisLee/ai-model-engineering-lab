"""Experiment tracking with MLflow.

Usage:
  export MLFLOW_TRACKING_URI=http://localhost:5000  # optional server
  python scripts/train.py  # automatically logs run

Or just run it — metrics are logged to ./mlruns/ by default.
Reference: https://mlflow.org
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from pathlib import Path

from ai_model_engineering_lab.utils.logging import setup_logging

logger = setup_logging()


class ExperimentTracker:
    """Thin wrapper around MLflow for training run tracking.

    Tracks: hyper-params, metrics per epoch, model artifact, config snapshot.
    """

    def __init__(
        self,
        experiment_name: str = "ai-model-engineering-lab",
        tracking_uri: Optional[str] = None,
    ):
        self._mlflow = None
        self._run = None
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        self._active = False

    def _ensure_mlflow(self):
        if self._mlflow is not None:
            return True
        try:
            import mlflow
            self._mlflow = mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            return True
        except ImportError:
            logger.warning("mlflow not installed. Run: pip install mlflow")
            return False

    def start_run(self, run_name: Optional[str] = None) -> bool:
        if not self._ensure_mlflow():
            return False
        self._run = self._mlflow.start_run(run_name=run_name)
        self._active = True
        return True

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._active:
            return
        flat = _flatten(params, prefix="")
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._active:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if not self._active:
            return
        self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        if not self._active or self._run is None:
            return
        self._mlflow.end_run()
        self._active = False


def _flatten(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Flatten nested dict for MLflow param logging."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items
