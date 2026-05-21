"""Точка входа: python -m src.train [--config path/to/config.yaml].

Запускает обучение baseline (TF-IDF + LR) и, при наличии флага --bert,
дополнительно fine-tune rubert-tiny2.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.dataset import dataset_to_dataframe, load_hf_dataset, save_sample
from src.models.baseline import train_baseline
from src.utils.config import get_artifacts_dir, get_data_dir, load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartFeedback — обучение моделей")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Путь к config.yaml",
    )
    parser.add_argument(
        "--bert",
        action="store_true",
        help="Дополнительно обучить rubert-tiny2",
    )
    parser.add_argument(
        "--save-sample",
        action="store_true",
        help="Сохранить выборку 500 строк в data/sample.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("log_level", "INFO"),
        log_file=config.get("log_file"),
    )

    artifacts_dir = get_artifacts_dir(config)
    data_dir = get_data_dir(config)

    ds = load_hf_dataset(cache_dir=config.get("hf_cache_dir"))
    df_train = dataset_to_dataframe(ds, split="train")
    df_test = dataset_to_dataframe(ds, split="test")

    logger.info("Train: %d | Test: %d", len(df_train), len(df_test))

    if args.save_sample:
        save_sample(df_train, data_dir / "sample.csv", n=500)

    logger.info("=== Обучение Baseline ===")
    baseline_cfg = config.get("baseline", {})
    baseline_results = train_baseline(df_train, df_test, baseline_cfg, artifacts_dir)
    logger.info(
        "Baseline F1-macro: %.4f  (цель >= 0.80)", baseline_results["f1_macro"]
    )

    if args.bert:
        from src.models.bert_model import train_bert

        logger.info("=== Fine-tuning rubert-tiny2 ===")
        bert_cfg = config.get("bert", {})
        bert_results = train_bert(df_train, df_test, bert_cfg, artifacts_dir)
        logger.info(
            "rubert-tiny2 F1-macro: %.4f  (цель >= 0.80)", bert_results["f1_macro"]
        )

    logger.info("Обучение завершено. Артефакты: %s", artifacts_dir)


if __name__ == "__main__":
    main()
