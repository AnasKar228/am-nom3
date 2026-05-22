"""Sanity-check тесты: проверяем полный пайплайн train→predict без реального датасета."""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import LABEL2ID, ID2LABEL, _clean_text
from src.models.baseline import build_pipeline, train_baseline


class TestLabelConsistency:
    """Проверка согласованности меток между модулями."""

    def test_label2id_and_id2label_roundtrip(self):
        for name, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == name, f"Несоответствие для класса {name}"

    def test_label_count_is_three(self):
        assert len(LABEL2ID) == 3
        assert len(ID2LABEL) == 3

    def test_no_duplicate_ids(self):
        ids = list(LABEL2ID.values())
        assert len(ids) == len(set(ids))


class TestBaselinePipelineSanity:
    """Sanity-check полного цикла обучения baseline."""

    @pytest.fixture
    def balanced_df(self):
        """DataFrame с равным числом примеров каждого класса."""
        rows = []
        texts_by_class = {
            0: ["отличный товар", "прекрасное качество", "рекомендую всем"],
            1: ["нормально в целом", "ничего особенного", "средний продукт"],
            2: ["плохое качество", "не рекомендую брать", "разочарован покупкой"],
        }
        for label_id, texts in texts_by_class.items():
            for text in texts:
                rows.append({"id": len(rows), "text": text,
                             "label": label_id,
                             "label_text": ID2LABEL[label_id]})
        return pd.DataFrame(rows)

    def test_pipeline_fits_without_error(self, balanced_df):
        pipe = build_pipeline(max_features=500)
        pipe.fit(balanced_df["text"], balanced_df["label"])

    def test_pipeline_predict_returns_correct_count(self, balanced_df):
        pipe = build_pipeline(max_features=500)
        pipe.fit(balanced_df["text"], balanced_df["label"])
        preds = pipe.predict(balanced_df["text"])
        assert len(preds) == len(balanced_df)

    def test_pipeline_predict_labels_in_valid_set(self, balanced_df):
        pipe = build_pipeline(max_features=500)
        pipe.fit(balanced_df["text"], balanced_df["label"])
        preds = pipe.predict(balanced_df["text"])
        assert set(preds).issubset({0, 1, 2})

    def test_proba_shape(self, balanced_df):
        pipe = build_pipeline(max_features=500)
        pipe.fit(balanced_df["text"], balanced_df["label"])
        proba = pipe.predict_proba(balanced_df["text"])
        assert proba.shape == (len(balanced_df), 3)

    def test_proba_rows_sum_to_one(self, balanced_df):
        pipe = build_pipeline(max_features=500)
        pipe.fit(balanced_df["text"], balanced_df["label"])
        proba = pipe.predict_proba(balanced_df["text"])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_train_baseline_metrics_saved(self, balanced_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(
                balanced_df, balanced_df,
                {"max_features": 500, "ngram_range": [1, 1], "C": 1.0},
                tmpdir,
            )
        assert 0.0 <= result["f1_macro"] <= 1.0
        assert 0.0 <= result["accuracy"] <= 1.0


class TestCleanTextSanity:
    """Граничные случаи предобработки текста."""

    @pytest.mark.parametrize("raw,expected", [
        ("привет\nмир", "привет мир"),
        ("двойной  пробел", "двойной пробел"),
        ("\t\tтаб", "таб"),
        ("   ", ""),
        ("нормальный текст", "нормальный текст"),
    ])
    def test_clean_text_cases(self, raw, expected):
        assert _clean_text(raw) == expected
