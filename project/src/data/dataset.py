"""Загрузка и предобработка датасета ai-forever/ru-reviews-classification."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)

DATASET_NAME = "ai-forever/ru-reviews-classification"
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _clean_text(text: str) -> str:
    """Базовая нормализация: убираем лишние пробелы и управляющие символы."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def load_hf_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    """Загружает датасет с Hugging Face.

    Returns:
        DatasetDict с ключами train / test.
    """
    logger.info("Загрузка датасета %s ...", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, cache_dir=cache_dir)
    logger.info("Датасет загружен: %s", {k: len(v) for k, v in ds.items()})
    return ds


def dataset_to_dataframe(ds: DatasetDict, split: str = "train") -> pd.DataFrame:
    """Конвертирует HF Dataset-сплит в pandas DataFrame с нормализованными текстами."""
    df = ds[split].to_pandas()
    df["text"] = df["text"].astype(str).apply(_clean_text)
    return df[["id", "text", "label", "label_text"]].copy()


def save_sample(df: pd.DataFrame, path: str | Path, n: int = 500) -> None:
    """Сохраняет выборку из n строк в CSV для быстрой демонстрации."""
    sample = df.sample(n=min(n, len(df)), random_state=42)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8")
    logger.info("Выборка из %d строк сохранена в %s", len(sample), path)
