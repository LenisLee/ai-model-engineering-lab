import pytest

from ai_model_engineering_lab.core.registry import Registry


class Dummy:
    pass


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry[str]("test")
        reg.register("a", "alpha")
        assert reg.get("a") == "alpha"

    def test_register_decorator(self):
        reg = Registry[type]("models")

        @reg.register("dummy")
        class DummyModel:
            pass

        assert reg.get("dummy") is DummyModel

    def test_get_missing_raises(self):
        reg = Registry[int]("ints")
        with pytest.raises(KeyError):
            reg.get("nope")

    def test_list(self):
        reg = Registry[str]("items")
        reg.register("x", "X")
        reg.register("y", "Y")
        assert set(reg.list()) == {"x", "y"}

    def test_contains(self):
        reg = Registry[int]("nums")
        reg.register("one", 1)
        assert "one" in reg
        assert "two" not in reg
