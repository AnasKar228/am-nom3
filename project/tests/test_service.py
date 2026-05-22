"""Тесты FastAPI-сервиса src/service/app.py.

Используется TestClient — без реального запуска uvicorn и без модели на диске.
Модель подменяется через monkeypatch, чтобы тесты были изолированы.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.service.app as app_module


@pytest.fixture(autouse=True)
def mock_predictor(monkeypatch):
    """Подменяем загрузку модели: baseline-заглушка, которая всегда возвращает 'positive'."""
    import numpy as np

    mock_pipe = MagicMock()
    mock_pipe.predict.return_value = [0]  # 0 → positive (по ID2LABEL)
    mock_pipe.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

    monkeypatch.setattr(app_module, "_model_type", "baseline")
    monkeypatch.setattr(app_module, "_predictor", mock_pipe)
    return mock_pipe


@pytest.fixture(scope="module")
def client():
    """TestClient FastAPI-приложения."""
    with TestClient(app_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_model_key(self, client):
        data = client.get("/health").json()
        assert "model" in data


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json={"text": "Отличный товар!"})
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"text": "хороший товар"}).json()
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_label_is_valid_class(self, client):
        data = client.post("/predict", json={"text": "хороший товар"}).json()
        assert data["label"] in {"positive", "neutral", "negative"}

    def test_predict_confidence_in_range(self, client):
        data = client.post("/predict", json={"text": "хороший товар"}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_three_keys(self, client):
        data = client.post("/predict", json={"text": "хороший товар"}).json()
        assert set(data["probabilities"].keys()) == {"positive", "neutral", "negative"}

    def test_predict_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json={"text": "хороший товар"}).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_empty_text_returns_422(self, client):
        """Pydantic должен отклонить пустой текст (min_length=1)."""
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_missing_text_field_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    def test_analyze_returns_200(self, client):
        resp = client.post("/analyze", json={"texts": ["хорошо", "плохо", "норм"]})
        assert resp.status_code == 200

    def test_analyze_response_schema(self, client):
        data = client.post("/analyze", json={"texts": ["отзыв 1", "отзыв 2"]}).json()
        assert "summary" in data
        assert "predictions" in data

    def test_analyze_summary_total_matches_input(self, client):
        texts = ["отзыв 1", "отзыв 2", "отзыв 3"]
        data = client.post("/analyze", json={"texts": texts}).json()
        assert data["summary"]["total"] == len(texts)

    def test_analyze_predictions_count_matches_input(self, client):
        texts = ["отзыв 1", "отзыв 2"]
        data = client.post("/analyze", json={"texts": texts}).json()
        assert len(data["predictions"]) == len(texts)

    def test_analyze_pct_sum_close_to_100(self, client):
        data = client.post("/analyze", json={"texts": ["a", "b", "c"]}).json()
        s = data["summary"]
        total_pct = s["positive_pct"] + s["neutral_pct"] + s["negative_pct"]
        assert abs(total_pct - 100.0) < 0.5

    def test_analyze_empty_list_returns_422(self, client):
        """Pydantic: min_length=1 для списка texts."""
        resp = client.post("/analyze", json={"texts": []})
        assert resp.status_code == 422

    def test_analyze_summary_counts_non_negative(self, client):
        data = client.post("/analyze", json={"texts": ["текст"]}).json()
        s = data["summary"]
        assert s["positive"] >= 0
        assert s["neutral"] >= 0
        assert s["negative"] >= 0
