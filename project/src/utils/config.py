"""Утилиты для загрузки конфигурации из YAML и переменных окружения."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Загружает YAML-конфиг и подставляет переменные окружения (${VAR})."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    for key, val in os.environ.items():
        raw = raw.replace(f"${{{key}}}", val)

    return yaml.safe_load(raw)


def get_artifacts_dir(config: dict[str, Any]) -> Path:
    path = Path(config.get("artifacts_dir", "artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(config: dict[str, Any]) -> Path:
    path = Path(config.get("data_dir", "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path
