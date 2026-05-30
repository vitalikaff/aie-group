"""Тесты FastAPI-сервиса.

Используем TestClient с подменённой моделью, чтобы не загружать тяжёлые веса.
"""
import io
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.config import load_config


class _DummyModel(torch.nn.Module):
    def __init__(self, image_size=(224, 224)):
        super().__init__()
        self.image_size = image_size
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, 1, *self.image_size, device=x.device)


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config()
    device = torch.device("cpu")
    dummy = _DummyModel(cfg.inference.image_size)

    def fake_load_model(cfg_in=None):
        return dummy, device, cfg_in or cfg

    with patch("src.service.app.load_model", side_effect=fake_load_model):
        from src.service.app import create_app
        app = create_app(cfg)
        with TestClient(app) as c:
            yield c


def _png_bytes() -> bytes:
    img = Image.fromarray(np.full((128, 128, 3), 127, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "architecture" in data
    assert "encoder" in data
    assert "description" in data
    assert "threshold" in data


def test_predict_returns_mask(client):
    r = client.post("/predict", files={"file": ("img.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert "mask_png_base64" in data
    assert 0.0 <= data["polyp_area_ratio"] <= 1.0
    assert data["image_size"] == [224, 224]


def test_predict_empty_file(client):
    r = client.post("/predict", files={"file": ("img.png", b"", "image/png")})
    assert r.status_code == 400


def test_predict_invalid_image(client):
    r = client.post("/predict", files={"file": ("img.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
