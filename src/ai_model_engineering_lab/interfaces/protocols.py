from typing import Protocol, Any, Dict
import torch


class ModelProtocol(Protocol):
    """Protocol for models — backbone + head as a unit."""

    def forward(self, **inputs: Any) -> Dict[str, torch.Tensor]: ...

    def predict(self, **inputs: Any) -> torch.Tensor: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "ModelProtocol": ...


class DatasetProtocol(Protocol):
    """Protocol for datasets — loading, preprocessing, tokenization."""

    def load(self) -> Any: ...

    def preprocess(self, **kwargs: Any) -> Any: ...

    def tokenize(self, tokenizer: Any, **kwargs: Any) -> Any: ...


class TrainerProtocol(Protocol):
    """Protocol for training loops."""

    def train(self) -> Dict[str, Any]: ...

    def evaluate(self) -> Dict[str, float]: ...

    def save_checkpoint(self, path: str) -> None: ...


class EvaluatorProtocol(Protocol):
    """Protocol for model evaluation."""

    def compute_metrics(self, predictions: Any, references: Any) -> Dict[str, float]: ...

    def evaluate(self, model: Any, dataset: Any) -> Dict[str, float]: ...


class PipelineProtocol(Protocol):
    """Protocol for end-to-end pipelines."""

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]: ...

    @property
    def name(self) -> str: ...
