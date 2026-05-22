"""Тесты модуля src/data/dataset.py."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import (
    LABEL2ID,
    ID2LABEL,
    _clean_text,
    dataset_to_dataframe,
    save_sample,
)


# ---------------------------------------------------------------------------
# Sanity-checks по константам
# ---------------------------------------------------------------------------

class TestConstants:
    def test_label2id_keys(self):
        """LABEL2ID должен содержать ровно три класса."""
        assert set(LABEL2ID.keys()) == {"positive", "neutral", "negative"}

    def test_id2label_is_inverse(self):
        """ID2LABEL — обратное отображение LABEL2ID."""
        for name, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == name

    def test_label_ids_are_0_1_2(self):
        """Значения LABEL2ID — ровно {0, 1, 2}."""
        assert set(LABEL2ID.values()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_newlines(self):
        assert "\n" not in _clean_text("строка1\nстрока2")

    def test_removes_tabs(self):
        assert "\t" not in _clean_text("слово\tеще")

    def test_collapses_spaces(self):
        result = _clean_text("много   пробелов   здесь")
        assert "  " not in result

    def test_strips_edges(self):
        result = _clean_text("  пробелы  ")
        assert result == "пробелы"

    def test_preserves_content(self):
        result = _clean_text("нормальный текст")
        assert result == "нормальный текст"

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_only_whitespace(self):
        assert _clean_text("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# dataset_to_dataframe (на mock-датасете без HF)
# ---------------------------------------------------------------------------

class TestDatasetToDataframe:
    def _make_mock_ds(self):
        """Создаём минимальный HF-подобный объект через словарь."""
        from datasets import Dataset, DatasetDict
        data = {
            "id": [1, 2, 3],
            "text": ["хорошо\nтовар", "плохо  продукт", "норм"],
            "label": [0, 2, 1],
            "label_text": ["positive", "negative", "neutral"],
        }
        ds = DatasetDict({"train": Dataset.from_dict(data)})
        return ds

    def test_returns_dataframe(self):
        ds = self._make_mock_ds()
        df = dataset_to_dataframe(ds, split="train")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        ds = self._make_mock_ds()
        df = dataset_to_dataframe(ds, split="train")
        assert set(df.columns) == {"id", "text", "label", "label_text"}

    def test_text_is_cleaned(self):
        """Переносы строк должны быть убраны после предобработки."""
        ds = self._make_mock_ds()
        df = dataset_to_dataframe(ds, split="train")
        for text in df["text"]:
            assert "\n" not in text
            assert "  " not in text  # двойных пробелов нет

    def test_row_count_preserved(self):
        ds = self._make_mock_ds()
        df = dataset_to_dataframe(ds, split="train")
        assert len(df) == 3

    def test_label_dtype(self):
        ds = self._make_mock_ds()
        df = dataset_to_dataframe(ds, split="train")
        assert pd.api.types.is_integer_dtype(df["label"])


# ---------------------------------------------------------------------------
# save_sample
# ---------------------------------------------------------------------------

class TestSaveSample:
    def test_creates_csv(self, tiny_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out" / "sample.csv"
            save_sample(tiny_df, path, n=3)
            assert path.exists()

    def test_csv_has_correct_columns(self, tiny_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            save_sample(tiny_df, path, n=3)
            result = pd.read_csv(path)
            assert set(result.columns) == {"id", "text", "label", "label_text"}

    def test_sample_not_larger_than_n(self, tiny_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            save_sample(tiny_df, path, n=2)
            result = pd.read_csv(path)
            assert len(result) <= 2

    def test_n_larger_than_df_takes_all(self, tiny_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            save_sample(tiny_df, path, n=10_000)
            result = pd.read_csv(path)
            assert len(result) == len(tiny_df)

    def test_creates_parent_dirs(self, tiny_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "sample.csv"
            save_sample(tiny_df, path)
            assert path.exists()
