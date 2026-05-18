import numpy as np

from ai_model_engineering_lab.evaluation.metrics import compute_classification_metrics, make_compute_metrics_fn


class TestMetrics:
    def test_perfect_prediction(self):
        preds = np.array([0, 1, 0, 1])
        refs = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(preds, refs)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_worst_prediction(self):
        preds = np.array([1, 0, 1, 0])
        refs = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(preds, refs)
        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0

    def test_compute_metrics_fn(self):
        fn = make_compute_metrics_fn()
        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1, 0, 1])
        metrics = fn((logits, labels))
        assert "accuracy" in metrics
        assert "f1" in metrics
