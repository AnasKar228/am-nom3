"""Общие фикстуры для всех тестов."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    """Минимальный DataFrame, имитирующий структуру датасета (без HF-запросов)."""
    rows = [
        {"id": 1, "text": "Отличный товар, очень доволен покупкой!", "label": 0, "label_text": "positive"},
        {"id": 2, "text": "Нормально, ничего особенного.", "label": 1, "label_text": "neutral"},
        {"id": 3, "text": "Ужасное качество, не рекомендую.", "label": 2, "label_text": "negative"},
        {"id": 4, "text": "Пришло быстро, упаковка целая.", "label": 0, "label_text": "positive"},
        {"id": 5, "text": "Не соответствует описанию на сайте.", "label": 2, "label_text": "negative"},
        {"id": 6, "text": "В целом неплохо, есть небольшие недостатки.", "label": 1, "label_text": "neutral"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_train(tiny_df):
    return tiny_df.iloc[:4].reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_test(tiny_df):
    return tiny_df.iloc[4:].reset_index(drop=True)


@pytest.fixture(scope="session")
def baseline_config() -> dict:
    return {"max_features": 1_000, "ngram_range": [1, 2], "C": 1.0}
