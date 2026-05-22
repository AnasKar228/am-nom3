"""Тесты модуля src/models/baseline.py."""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.baseline import build_pipeline, load_baseline, train_baseline


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline(self):
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_steps(self):
        pipe = build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "tfidf" in step_names
        assert "clf" in step_names

    def test_custom_max_features(self):
        pipe = build_pipeline(max_features=100)
        assert pipe.named_steps["tfidf"].max_features == 100

    def test_custom_ngram_range(self):
        pipe = build_pipeline(ngram_range=(1, 3))
        assert pipe.named_steps["tfidf"].ngram_range == (1, 3)

    def test_custom_C(self):
        pipe = build_pipeline(C=0.5)
        assert pipe.named_steps["clf"].C == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# train_baseline
# ---------------------------------------------------------------------------

class TestTrainBaseline:
    def test_returns_dict_with_metrics(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
        assert "f1_macro" in result
        assert "accuracy" in result
        assert "report" in result

    def test_f1_macro_in_valid_range(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
        assert 0.0 <= result["f1_macro"] <= 1.0

    def test_accuracy_in_valid_range(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_saves_pkl_file(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            assert (Path(tmpdir) / "baseline_pipeline.pkl").exists()

    def test_saves_metrics_json(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            metrics_path = Path(tmpdir) / "baseline_metrics.json"
            assert metrics_path.exists()

    def test_metrics_json_valid(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            with open(Path(tmpdir) / "baseline_metrics.json") as f:
                data = json.load(f)
            assert "accuracy" in data
            assert "f1_macro" in data
            assert "report" in data

    def test_report_has_three_classes(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
        report_keys = set(result["report"].keys())
        for cls in {"positive", "neutral", "negative"}:
            assert cls in report_keys

    def test_model_path_in_result(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
        assert "model_path" in result
        assert Path(result["model_path"]).exists()

    def test_creates_artifacts_dir_if_missing(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "nested" / "dir"
            train_baseline(tiny_train, tiny_test, baseline_config, new_dir)
            assert new_dir.exists()


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------

class TestLoadBaseline:
    def test_loads_saved_pipeline(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            loaded = load_baseline(Path(tmpdir) / "baseline_pipeline.pkl")
        assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_can_predict(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            loaded = load_baseline(Path(tmpdir) / "baseline_pipeline.pkl")
        preds = loaded.predict(["хороший товар", "плохая вещь"])
        assert len(preds) == 2

    def test_loaded_pipeline_proba_shape(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            loaded = load_baseline(Path(tmpdir) / "baseline_pipeline.pkl")
        proba = loaded.predict_proba(["хороший товар"])
        assert proba.shape == (1, 3)  # 3 класса

    def test_proba_sums_to_one(self, tiny_train, tiny_test, baseline_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_baseline(tiny_train, tiny_test, baseline_config, tmpdir)
            loaded = load_baseline(Path(tmpdir) / "baseline_pipeline.pkl")
        import numpy as np
        proba = loaded.predict_proba(["нормальный отзыв"])
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/path/model.pkl")
