"""Baseline-модель: TF-IDF + LogisticRegression."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_pipeline(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
) -> Pipeline:
    """Создаёт sklearn Pipeline: TF-IDF → LogisticRegression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    analyzer="word",
                    token_pattern=r"\b\w+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="multinomial",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    config: dict[str, Any],
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    """Обучает baseline и возвращает словарь с метриками."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(
        max_features=config.get("max_features", 50_000),
        ngram_range=tuple(config.get("ngram_range", [1, 2])),
        C=config.get("C", 1.0),
    )

    logger.info("Обучение baseline TF-IDF + LR на %d примерах ...", len(df_train))
    pipeline.fit(df_train["text"], df_train["label"])

    y_pred = pipeline.predict(df_test["text"])
    y_true = df_test["label"].values

    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["positive", "neutral", "negative"],
        output_dict=True,
    )
    logger.info("Baseline F1-macro = %.4f", f1)

    model_path = artifacts_dir / "baseline_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    metrics_path = artifacts_dir / "baseline_metrics.json"
    metrics = {"f1_macro": f1, "report": report}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Модель сохранена: %s", model_path)
    logger.info("Метрики сохранены: %s", metrics_path)
    return {"f1_macro": f1, "report": report, "model_path": str(model_path)}


def load_baseline(path: str | Path) -> Pipeline:
    """Загружает сохранённый baseline из pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
