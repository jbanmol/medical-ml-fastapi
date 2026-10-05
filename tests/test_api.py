import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "example_request.json").read_text())


@pytest.fixture
def client():
    with TestClient(main.app) as connection:
        yield connection


def test_home(client):
    assert client.get("/").status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_prediction(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == {0: "malignant", 1: "benign"}[body["prediction_code"]]
    assert body["probability"] == body["class_probabilities"][body["prediction"]]
    assert sum(body["class_probabilities"].values()) == pytest.approx(1, abs=0.0001)
    assert "educational" in body["disclaimer"]


def test_missing_fields(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.mark.parametrize("value", [-1, "invalid", True, None])
def test_invalid_values(client, value):
    assert client.post("/predict", json={**SAMPLE, "mean_radius": value}).status_code == 422


def test_extra_field(client):
    assert client.post("/predict", json={**SAMPLE, "unknown": 2}).status_code == 422


def test_invalid_json(client):
    assert client.post("/predict", content="{broken", headers={"Content-Type": "application/json"}).status_code == 422


def test_docs(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["components"]["schemas"]["PredictionRequest"]["examples"] == [SAMPLE]


@pytest.mark.parametrize("contents", [None, b"not a pickle"])
def test_unavailable_model(monkeypatch, tmp_path, contents):
    path = tmp_path / "model.pkl"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(main, "MODEL_PATH", path)
    with TestClient(main.app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "error", "model_loaded": False}
        assert client.post("/predict", json=SAMPLE).status_code == 503


def test_prediction_failure(client, monkeypatch):
    def fail(_):
        raise RuntimeError("private diagnostic details")
    monkeypatch.setattr(main.app.state.model, "predict", fail)
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
