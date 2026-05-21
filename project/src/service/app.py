"""FastAPI-сервис SmartFeedback.

Эндпоинты:
  GET  /health       — health-check
  POST /predict      — тональность одного отзыва
  POST /analyze      — сводная статистика по батчу отзывов
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Текст отзыва")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)


class AnalyzeSummary(BaseModel):
    total: int
    positive: int
    neutral: int
    negative: int
    positive_pct: float
    neutral_pct: float
    negative_pct: float


class AnalyzeResponse(BaseModel):
    summary: AnalyzeSummary
    predictions: list[PredictResponse]


def _load_predictor():
    """Загружает модель по приоритету: BERT → baseline."""
    bert_path = Path(os.getenv("BERT_MODEL_PATH", "artifacts/rubert_tiny2/final"))
    baseline_path = Path(
        os.getenv("BASELINE_MODEL_PATH", "artifacts/baseline_pipeline.pkl")
    )

    if bert_path.exists():
        from src.models.bert_model import BertPredictor

        logger.info("Загружается BERT-модель из %s", bert_path)
        return "bert", BertPredictor(bert_path)

    if baseline_path.exists():
        logger.info("Загружается baseline из %s", baseline_path)
        with open(baseline_path, "rb") as f:
            pipeline = pickle.load(f)
        return "baseline", pipeline

    raise RuntimeError(
        "Модель не найдена. Запустите обучение: python -m src.train"
    )


app = FastAPI(
    title="SmartFeedback",
    description="Анализатор тональности отзывов Wildberries",
    version="1.0.0",
)

_model_type: str = "unknown"
_predictor = None


@app.on_event("startup")
def startup_event():
    global _model_type, _predictor
    _model_type, _predictor = _load_predictor()
    logger.info("Сервис запущен, используется модель: %s", _model_type)


def _predict_one(text: str) -> dict:
    if _model_type == "bert":
        return _predictor.predict([text])[0]
    proba = _predictor.predict_proba([text])[0]
    label_id = int(proba.argmax())
    id2label = {0: "positive", 1: "neutral", 2: "negative"}
    return {
        "label": id2label[label_id],
        "confidence": round(float(proba[label_id]), 4),
        "probabilities": {id2label[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


@app.get("/health", tags=["monitoring"])
def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "model": _model_type}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """Классифицирует тональность одного отзыва."""
    t0 = time.perf_counter()
    try:
        result = _predict_one(request.text)
    except Exception as e:
        logger.exception("Ошибка инференса: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка модели")
    elapsed = time.perf_counter() - t0
    logger.info(
        "predict | label=%s conf=%.3f latency=%.3fs",
        result["label"], result["confidence"], elapsed,
    )
    return result


@app.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
def analyze(request: AnalyzeRequest):
    """Классифицирует батч отзывов и возвращает сводную статистику."""
    t0 = time.perf_counter()
    try:
        if _model_type == "bert":
            predictions = _predictor.predict(request.texts)
        else:
            predictions = [_predict_one(t) for t in request.texts]
    except Exception as e:
        logger.exception("Ошибка батч-инференса: %s", e)
        raise HTTPException(status_code=500, detail="Ошибка модели")

    elapsed = time.perf_counter() - t0
    counts = Counter(p["label"] for p in predictions)
    total = len(predictions)
    summary = AnalyzeSummary(
        total=total,
        positive=counts.get("positive", 0),
        neutral=counts.get("neutral", 0),
        negative=counts.get("negative", 0),
        positive_pct=round(counts.get("positive", 0) / total * 100, 1),
        neutral_pct=round(counts.get("neutral", 0) / total * 100, 1),
        negative_pct=round(counts.get("negative", 0) / total * 100, 1),
    )
    logger.info(
        "analyze | total=%d positive=%.1f%% negative=%.1f%% latency=%.3fs",
        total, summary.positive_pct, summary.negative_pct, elapsed,
    )
    return AnalyzeResponse(
        summary=summary,
        predictions=[PredictResponse(**p) for p in predictions],
    )
