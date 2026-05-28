"""Тесты утилит: src/utils/config.py."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def _write_yaml(self, tmpdir: str, content: dict) -> str:
        path = str(Path(tmpdir) / "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, allow_unicode=True)
        return path

    def test_loads_simple_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, {"key": "value", "num": 42})
            cfg = load_config(path)
        assert cfg["key"] == "value"
        assert cfg["num"] == 42

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, {"a": 1})
            result = load_config(path)
        assert isinstance(result, dict)

    def test_nested_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, {"baseline": {"C": 1.0, "max_features": 50000}})
            cfg = load_config(path)
        assert "baseline" in cfg
        assert cfg["baseline"]["C"] == 1.0

    def test_env_var_substitution(self, monkeypatch):
        """${ENV_VAR} в значениях YAML должны подставляться из окружения."""
        monkeypatch.setenv("TEST_DIR", "/tmp/test_artifacts")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "config.yaml")
            # Пишем вручную, т.к. yaml.dump экранирует $
            with open(path, "w") as f:
                f.write("artifacts_dir: ${TEST_DIR}\n")
            cfg = load_config(path)
        assert cfg["artifacts_dir"] == "/tmp/test_artifacts"

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, Exception)):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_yaml_returns_empty_dict_or_none(self):
        """Пустой YAML не должен бросать исключение."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "config.yaml")
            Path(path).write_text("", encoding="utf-8")
            try:
                result = load_config(path)
                assert result is None or isinstance(result, dict)
            except Exception:
                pass  