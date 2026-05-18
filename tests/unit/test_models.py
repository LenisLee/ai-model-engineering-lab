import pytest

from ai_model_engineering_lab.models.factory import create_model, MODEL_ARCHITECTURES


class TestModelFactory:
    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("bert-base-uncased", num_labels=2, architecture="invalid")

    def test_list_architectures(self):
        archs = list(MODEL_ARCHITECTURES.keys())
        assert "sequence_classification" in archs
        assert "causal_lm" in archs
