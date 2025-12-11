# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
## Добавления

# eda-cli

Мини-CLI для первичного EDA CSV-файлов.

## Команды

### 1. Обзор датасета

```bash
uv run eda-cli overview data/example.csv
#Печатает в терминал:количество строк и столбцов;табличку по колонкам (типы, пропуски, базовые статистики).

## Полный отчет
uv run eda-cli report data/example.csv --out-dir reports/example
Генерирует:

report.md — основной текстовый отчёт;

summary.csv — сводная информация по колонкам;

missing.csv — статистика пропусков;

correlation.csv — корреляционная матрица;

top_categories/*.csv — top-k категорий для категориальных признаков;

hist_*.png, missing_matrix.png, correlation_heatmap.png — графики.

##Параметры команды report

--max-hist-columns #— максимум числовых колонок для гистограмм (по умолчанию 6).

--top-k-categories #— сколько top-значений сохранять для категориальных признаков (по умолчанию 5).

--min-missing-share #— порог доли пропусков (от 0 до 1), выше которого колонка считается проблемной и попадает в отдельный список в отчёте (по умолчанию 0.2 = 20%).

--title #— заголовок отчёта (# ... в начале report.md).

##Пример запуска
uv run eda-cli report data/example.csv --out-dir reports/example --max-hist-columns 4 --top-k-categories 3 --min-missing-share 0.05 --title "Отчёт по example.csv"

#Доп задание.Вар А. Команда head
head       #Вывести первые n строк датасета.

#Пример запуска
# Показать первые 5 строк (по умолчанию n=5)
uv run eda-cli head data/example.csv

# Показать первые 10 строк
uv run eda-cli head data/example.csv --n 10

# Если вдруг другой разделитель или кодировка
uv run eda-cli head data/some_data.csv --sep ";" --encoding "cp1251"

### HTTP-сервис качества датасета (S04)

Сервис поднимается командой:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000

Документация (Swagger UI доступен по адресу http://127.0.0.1:8000/docs).

Эндпоинт: POST /quality-flags-from-csv

Принимает CSV-файл и возвращает полный набор флагов качества, рассчитанный функцией
compute_quality_flags из HW03.

## Запрос:
Метод: POST

Путь: /quality-flags-from-csv
Content-Type: multipart/form-data

Параметры:
file — CSV-файл с данными.

Пример запроса (curl):
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/example.csv;type=text/csv"
Ответ (JSON):
{
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "max_missing_share": 0.05,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": true,
    "has_suspicious_id_duplicates": true,
    "...": "другие флаги из compute_quality_flags"
  },
  "dataset_shape": {
    "n_rows": 36,
    "n_cols": 14
  }
}
