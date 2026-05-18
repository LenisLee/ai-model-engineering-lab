import torch

from ai_model_engineering_lab.utils.device import get_device


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")
