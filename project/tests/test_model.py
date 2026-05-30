"""Sanity-тесты архитектур SwinUNet и UNet: forward даёт корректную форму."""
import pytest
import torch

from src.models import SwinUNet, UNet, build_model


def test_unet_forward_shape():
    model = UNet(num_classes=1, image_size=(224, 224))
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 224, 224)


def test_unet_parameter_count_in_range():
    """U-Net должен быть лёгким (~7-8M параметров)."""
    model = UNet()
    n = sum(p.numel() for p in model.parameters())
    assert 5e6 < n < 1e7


def test_build_model_factory_dispatch():
    m = build_model("unet")
    assert isinstance(m, UNet)
    with pytest.raises(ValueError):
        build_model("nope")
    with pytest.raises(ValueError):
        build_model("swin_unet")  # без encoder_name


@pytest.mark.slow
def test_swin_unet_forward_shape():
    model = SwinUNet(encoder_name="swin_small_patch4_window7_224", pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 224, 224)


def test_double_conv_block_shapes():
    block = SwinUNet._double_conv(16, 32)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == (2, 32, 8, 8)
