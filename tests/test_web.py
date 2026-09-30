"""Exercise HTTP wiring with a tiny deterministic model, without real weights."""

import io
from pathlib import Path

import pytest
import server
import torch
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.arange(208, dtype=torch.float32))

        def forward(self, images):
            return self.bias.unsqueeze(0).expand(images.shape[0], -1)

    def fake_load(**kwargs):
        return TinyModel().eval(), {
            "model_name": kwargs["model_name"], "num_classes": 208,
            "device": "cpu", "loaded": 1, "total": 1, "head_warning": "",
        }

    monkeypatch.setattr(server, "load_model", fake_load)
    monkeypatch.setattr(server, "CLASS_CSV", Path(__file__).resolve().parents[1] / "data/metadata/class_names.csv")
    with TestClient(server.app) as test_client:
        yield test_client


def test_health_labels_and_static_assets(client):
    assert client.get("/api/health").json()["num_classes"] == 208
    assert len(client.get("/api/classes").json()["classes"]) == 208
    assert client.get("/").status_code == 200
    for name in ("script.js", "style.css"):
        assert client.get(f"/static/{name}").status_code == 200


def test_image_upload_returns_top_k(client):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    response = client.post("/api/predict?top_k=3", files={"file": ("food.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["top_k"] == 3
    assert [item["class_id"] for item in result["predictions"]] == [207, 206, 205]


def test_invalid_uploads_are_rejected(client):
    response = client.post("/api/predict", files={"file": ("x.txt", b"text", "text/plain")})
    assert response.status_code == 415
    response = client.post("/api/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400
