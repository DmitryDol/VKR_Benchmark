# Результаты и артефакты

Документ объясняет, **где** хранятся результаты бенчмаркинга и публикационные артефакты и **как**
их интерпретировать. Структуру метрик см. в [metrics.md](metrics.md); скрипты, которые всё это
генерируют, — в [scripts.md](scripts.md).

## Объём корпуса: 35 валидных конфигураций

Через конвейер проведены **4 модели** (RT-DETR, YOLO11l, YOLO26l, RF-DETR-L). Полная сетка —
4 × 10 стадий = 40, но 5 стадий RF-DETR (`5_trt_int8_*` и `6_trt_mixed_*`) исключены из-за «отката»
INT8→FP16 (см. [models.md](models.md#rf-detr-l-и-эффект-отката-int8)). Итого — **35 валидных
конфигураций**, на которых строятся все дипломные артефакты.

## Карта файлов результатов

| Путь | Что это |
|------|---------|
| [`results/results.csv`](../results/results.csv) | Единая плоская таблица всех прогонов (35 колонок, см. [metrics.md](metrics.md)) |
| [`results/results.json`](../results/results.json) | То же в JSON (агрегат по моделям) |
| `results/<model>/<run_id>/<stage>.csv` | Постадийный CSV одной конфигурации |
| `results/<model>/<run_id>/<stage>.json` | Постадийный JSON (включает `per_class_ap` по 80 классам) |
| `results/<model>/<run_id>/int8_best_calibrator.json` | Победитель среди INT8-калибраторов (для стадии 6) |
| `results/<model>/<run_id>/summary.{txt,md}` | Человекочитаемая сводка по модели (после `merge`) |
| [`results/per_class/`](../results/per_class/) | 8 CSV: топ-10 по просадке AP и топ-10 по частоте, на каждую модель |
| `results/confusion_80/` | Матрицы ошибок 80×80 (приложение), PNG на каждую конфигурацию |
| `cache/predictions/coco_dt_<model>_<stage>.json` | Кэш предсказаний (вход для матриц ошибок и per-class) |

> Каталоги прогонов (`<run_id>`) в текущем корпусе: `quant` — для RT-DETR/YOLO11l/YOLO26l,
> `rfdetr_v1` — для RF-DETR-L.

## Карта визуальных артефактов (`media/`)

| Путь | Что это | Генератор |
|------|---------|-----------|
| [`media/pareto/pareto_best_configs_v2.png`](../media/pareto/pareto_best_configs_v2.png) | Pareto latency↔mAP, рабочая зона 6–17 мс (главный график) | `plots_v2.py` |
| [`media/pareto/pareto_full_range_v2.png`](../media/pareto/pareto_full_range_v2.png) | Pareto на полном диапазоне 0–40 мс | `plots_v2.py` |
| [`media/pareto/precision_sweep_combined_v2.png`](../media/pareto/precision_sweep_combined_v2.png) | Свип точности (mAP + latency), главный для защиты | `plots_v2.py` |
| `media/pareto/pareto_latency_map.png`, `precision_sweep.png` | Версии v1 тех же графиков | `pareto_curve.py`, `precision_sweep.py` |
| [`media/confusion_12/`](../media/confusion_12/) | 35 матриц ошибок 12×12 (супер-категории, аннотированные) | `build_confusion.py` |
| [`media/qualitative/`](../media/qualitative/) | 12 коллажей качественной детекции (4 модели × 3 сцены) | `qualitative_examples.py` |
| [`media/coco_val2017_samples.png`](../media/coco_val2017_samples.png) | 3×3 коллаж образцов COCO с GT-боксами (hero-фигура) | `coco_collage.py` |
| [`media/per_class_md/`](../media/per_class_md/) | Markdown-таблицы per-class AP | `per_class_summary.py` |
| `media/video/` | Демо-видео с FPS-оверлеем (если задан `data/demo.mp4`) | `realtime_demo.py` |
| [`media/pres/`](../media/pres/) | Презентация ВКР (`.pptx` / `.pdf`) | — |

## Как читать артефакты

### Pareto-кривая latency ↔ mAP

![Pareto best configs](../media/pareto/pareto_best_configs_v2.png)

Каждая точка — конфигурация `(модель, стадия)`: X — `latency_total_ms`, Y — `map_50_95`. Цвет
кодирует модель, форма маркера — семейство стадии. Линия Парето-фронта соединяет недоминируемые
точки (быстрее и/или точнее). Идеал — левый верхний угол (низкая задержка, высокий mAP). Палитра и
маркеры подобраны так, чтобы график читался и в чёрно-белой печати. Версия `full_range` показывает
весь диапазон, `best_configs` — приближение к рабочей зоне 6–17 мс с подписями ключевых точек.

### Свип точности (precision sweep)

![Precision sweep](../media/pareto/precision_sweep_combined_v2.png)

Две панели для всех 4 моделей по категориальной оси точности
`FP32 → TF32 → FP16 → BF16 → INT8 → Mixed`: левая — mAP@[0.5:0.95], правая — latency. Для категорий
с несколькими вариантами (INT8 — 3 калибратора, Mixed — 2 стратегии) берётся вариант с наибольшим
mAP. На графике отмечается «оптимум снижения точности» — последний шаг, где относительный выигрыш в
скорости ещё превышает относительную потерю точности. У RF-DETR линии заканчиваются на BF16
(INT8/Mixed исключены).

### Матрицы ошибок (confusion matrices)

[`media/confusion_12/`](../media/confusion_12/) — агрегированные 12×12 матрицы по супер-категориям
COCO (`person`, `vehicle`, … + `background`), для основного текста; `results/confusion_80/` — полные
80×80 для приложения. Строятся жадным сопоставлением по IoU (порог **0.5**, confidence **0.25**):
совпадение → ячейка `(GT-класс, pred-класс)`, несопоставленный GT → столбец `background`,
несопоставленное предсказание → строка `background`. Матрицы **построчно нормированы** (сумма строки
= 1): диагональ — доля верных классификаций, внедиагональные ячейки — типичные путаницы. Имя файла —
`<model>_<stage>.png`.

### Таблицы per-class AP

[`results/per_class/`](../results/per_class/) (CSV) и [`media/per_class_md/`](../media/per_class_md/)
(Markdown): на каждую модель — две таблицы:

- `<model>_drop_top10` — 10 классов с наибольшей просадкой AP при квантовании;
- `<model>_freq_top10` — 10 самых частых классов (по числу GT-объектов `n_gt`).

Колонки включают `class_name`, `n_gt` и AP по стадиям. Полные значения per-class AP по 80 классам
лежат в постадийных JSON (`results/<model>/<run_id>/<stage>.json`, поле `per_class_ap`).

### Qualitative-примеры

[`media/qualitative/`](../media/qualitative/) — по 3 сценария на модель: `dense` (плотная сцена),
`occluded` (перекрытия), `large_single` (крупный одиночный объект). Каждый коллаж сравнивает режимы
точности (для не-RFDETR: PyTorch FP32 / ONNX / TF32 / FP16 / BF16 / лучший INT8 / худший INT8 /
лучший Mixed; для RF-DETR — только 5 валидных стадий) с подписью mAP и latency. Источник — кэш
предсказаний, без повторного инференса.

## Воспроизводимость

Артефакты детерминированы: порядок данных фиксирован (`sorted(image_ids)`, без shuffle), seeds
выставлены в скриптах, PNG-метаданные вычищаются для стабильного sha256. Повторный запуск
генераторов на том же `results/`/`cache/` даёт идентичные файлы.

> **Бэкафилл кэша.** Часть артефактов зависит от `cache/predictions/`. Если кэш неполон, выполните
> один долгий проход `uv run python scripts/build_per_class_ap.py --live` (см.
> [STATE.md](../.planning/STATE.md) → Operator Next Steps), затем перезапустите генераторы
> матриц/таблиц/коллажей.

## См. также

- [metrics.md](metrics.md) — что означает каждая колонка результатов.
- [scripts.md](scripts.md) — как сгенерировать каждый артефакт.
- [pipeline.md](pipeline.md) — почему у RF-DETR только 5 стадий.
