"""Fine-tuning rubert-tiny2 для классификации тональности."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "cointegrated/rubert-tiny2"

# ai-forever/ru-reviews-classification label encoding:
# 0 = negative, 1 = neutral, 2 = positive
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["negative", "neutral", "positive"]


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),  
    }


def tokenize_dataset(df, tokenizer, max_length: int = 128) -> Dataset:
    """Токенизирует DataFrame и возвращает HF Dataset."""
    hf_ds = Dataset.from_pandas(
        df[["text", "label"]].rename(columns={"label": "labels"})
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return hf_ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def train_bert(
    df_train,
    df_test,
    config: dict[str, Any],
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    """Fine-tune rubert-tiny2, сохраняет модель и метрики."""
    artifacts_dir = Path(artifacts_dir)
    model_dir = artifacts_dir / "rubert_tiny2"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    max_length = config.get("max_length", 128)
    train_ds = tokenize_dataset(df_train, tokenizer, max_length)
    test_ds = tokenize_dataset(df_test, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 64),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=0.01,
        warmup_ratio=config.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    logger.info("Fine-tuning %s на %d примерах ...", MODEL_NAME, len(df_train))
    trainer.train()

    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = df_test["label"].values

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()  
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
    )
    logger.info("rubert-tiny2  F1-macro = %.4f  Accuracy = %.4f", f1, acc)

    trainer.save_model(str(model_dir / "final"))
    tokenizer.save_pretrained(str(model_dir / "final"))

    metrics_path = artifacts_dir / "bert_metrics.json"
    metrics = {
        "f1_macro": f1,
        "accuracy": acc,
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": cm,
        },
        "report": report,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Модель сохранена: %s", model_dir / "final")
    logger.info("Метрики сохранены: %s", metrics_path)
    return {
        "f1_macro": f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "model_path": str(model_dir / "final"),
    }


class BertPredictor:
    """Обёртка для инференса дообученного rubert-tiny2."""

    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str], max_length: int = 128) -> list[dict]:
        """Возвращает список {label, confidence, probabilities} для каждого текста."""
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            label_id = int(probs.argmax())
            results.append(
                {
                    "label": ID2LABEL[label_id],
                    "confidence": round(float(probs[label_id]), 4),
                    "probabilities": {
                        ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
                    },
                }
            )
        return results
