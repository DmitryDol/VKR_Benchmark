# Модели и адаптеры

Документ описывает поддерживаемые детекторы, их адаптеры
([`src/benchmark/models/`](../src/benchmark/models/)), особенности препроцессинга и парсинга
выходов, маппинг классов COCO, а также инструкцию по добавлению новой модели. Контракт адаптера —
паттерн `ModelAdapter`, см. [architecture.md](architecture.md#strategy-через-protocol--modeladapter).

## Сводка

| Модель | `--model` | Источник весов | Backbone | Вход | Классы | Постпроцессинг | Семейство¹ | Статус |
|--------|-----------|----------------|----------|:----:|:------:|----------------|:----------:|:------:|
| RT-DETR (r50vd) | `rt-detr` | HF `PekingU/rtdetr_r50vd` | ResNet-50 | 640×640 | COCO-80 | sigmoid + порог + per-query argmax | `detr` | ✅ |
| YOLO11l | `yolo11l` | Ultralytics `.pt` | YOLO11 | 640×640 | COCO-80 | letterbox + NMS (IoU 0.7) | `yolo` | ✅ |
| YOLO26l | `yolo26l` | Ultralytics `.pt` | YOLO26 | 640×640 | COCO-80 | letterbox + NMS-free (end2end) | `yolo` | ✅ |
| RF-DETR-L | `rfdetr-l` | `rfdetr.RFDETRLarge` | DINOv2 + LWDETR | 704×704 | COCO-91 | sigmoid + top-k по (queries×classes) | `rfdetr` | ✅² |
| D-FINE | — | — | — | — | — | — | `detr` | 🔜 |
| DEIMv2 | — | — | — | — | — | — | `detr` | 🔜 |

¹ Семейство определяет способ подсчёта MACs/FLOPs (см. [`utils/macs.py`](../src/benchmark/utils/macs.py)):
`detr`/`rfdetr` → `calflops`, `yolo` → нативный `model.info()`.
² INT8/Mixed-стадии исключены — см. [ниже](#rf-detr-l-и-эффект-отката-int8).

Реестр моделей задан в [`cli.py`](../src/benchmark/cli.py) (`MODEL_REGISTRY`): для каждой модели —
путь к весам, путь к упрощённому ONNX и семейство.

## Маппинг классов COCO-80 ↔ COCO-91

COCO исторически нумерует категории от 1 до 90 с 11 «дырами» (отсутствуют id 12, 26, 29, 30, 45, 66,
68, 69, 71, 83, 91) — это «COCO-91». Большинство моделей предсказывают плотный диапазон из 80 классов
(«COCO-80»). Преобразование задано в [`data/coco_loader.py`](../src/benchmark/data/coco_loader.py):
словари `COCO_91_TO_80` и `COCO_80_TO_91`. Итоговые `Detection.labels` **всегда** содержат
COCO-91 category_id — именно их ожидает COCOeval.

- **RT-DETR** предсказывает 80 классов и переводит индекс 0–79 → COCO-91 через явный LUT
  `_COCO80_LUT` (хардкод в адаптере, чтобы не зависеть от загрузчика).
- **YOLO** возвращает классы 0–79 и переводит их через `COCO_80_TO_91`.
- **RF-DETR** работает прямо в COCO-91 — LUT не нужен (см. ниже).

---

## RT-DETR — [`rtdetr_adapter.py`](../src/benchmark/models/rtdetr_adapter.py)

- **Модель:** `PekingU/rtdetr_r50vd` (HuggingFace `RTDetrForObjectDetection`, backbone ResNet-50).
- **Вход:** 640×640. **Препроцессинг:** общий stretch-resize (без letterbox), нормализация в [0, 1].
- **Выход:** логиты `(1, 300, 80)` и боксы `(1, 300, 4)`. **Фона нет** — 80 классов напрямую.
- **Парсинг (`parse_outputs`):** `sigmoid` по логитам → max по классам → порог `score_threshold`
  (0.001) → перевод нормализованных `cx,cy,w,h` в пиксельные `x1y1x2y2` исходного изображения →
  индекс 0–79 в COCO-91 через `_COCO80_LUT`.
- **ONNX-экспорт:** обёртка `RTDetrONNXWrapper` превращает HF-модель (kwargs + dataclass-выход) в
  трассируемый модуль, возвращающий кортеж `(logits, pred_boxes)`.

## YOLO11 / YOLO26 — [`yolo_adapter.py`](../src/benchmark/models/yolo_adapter.py)

- **Модель:** Ultralytics YOLO; единый `YOLOAdapter(is_nms_free=…)` обслуживает оба варианта.
  CLI создаёт его с `is_nms_free=True` для `yolo26l` и `False` для `yolo11l`.
- **Вход:** 640×640. **Препроцессинг — letterbox** (масштаб по длинной стороне + центрирующий
  паддинг значением 114), нормализация в [0, 1]. Это критично: YOLO обучались с letterbox, и
  калибровка/инференс должны совпадать с обучением, иначе mAP занижается на 2–3 пункта. Адаптер
  **обязан** реализовывать `preprocess` (в отличие от опционального контракта протокола).
- **Постпроцессинг:**
  - **YOLO11 (NMS):** `ultralytics.utils.nms.non_max_suppression` (conf = `score_threshold`,
    IoU = 0.7, nc = 80) → обратное letterbox-преобразование боксов в координаты оригинала.
  - **YOLO26 (NMS-free):** выход уже в виде `[x1,y1,x2,y2,conf,cls]`, применяется только порог по
    confidence и то же обратное letterbox.
  - Классы 0–79 переводятся в COCO-91 через `COCO_80_TO_91`.
- **Поддержка путей:** парсер принимает и `torch.Tensor` (PyTorch/TensorRT), и `list[np.ndarray]`
  (ONNX Runtime).

## RF-DETR-L — [`rfdetr_adapter.py`](../src/benchmark/models/rfdetr_adapter.py)

- **Модель:** `rfdetr.RFDETRLarge` (backbone DINOv2 + декодер LWDETR). Веса (~150 МБ,
  `rf-detr-large-2026.pth`) скачиваются вендором при первой загрузке; каталог `weights/rfdetr-l/`
  нужен лишь для единообразия раскладки.
- **Вход:** **704×704** (нативное разрешение). **Препроцессинг:** прямой resize (без letterbox —
  важно для выравнивания патчей DINOv2) + нормализация по ImageNet (mean/std).
- **Выход:** `pred_logits` `(1, 300, 91)` и `pred_boxes` `(1, 300, 4)`. Классы — **COCO-91**:
  слот 0 — N/A (нет COCO id = 0), слоты 1..89 — COCO-id, слот **90 — фон** (DETR-конвенция).
- **Парсинг:** `sigmoid` → **top-k (300) по плоскому пространству (queries × classes)** (одна
  query может дать несколько детекций разных классов — в отличие от per-query argmax у RT-DETR) →
  декомпозиция плоского индекса в `(query_idx, class_idx)` → отбрасывание `class_idx == 0` и
  `class_idx == 90` → порог → денормализация боксов. LUT не нужен — метки уже COCO-91.
- **Особенность путей:** порядок выходов ONNX определяется **по форме**, а не по индексу
  (RF-DETR экспортирует `[dets, labels]`, RT-DETR — `[logits, pred_boxes]`).

### RF-DETR-L и эффект «отката» INT8

При сборке INT8/Mixed-движков для RF-DETR авто-тюнер TensorRT 10.16 на трансформерном графе выбирает
почти исключительно FP16-ядра — доля реальных INT8-слоёв составляет ≈ 0–0.78%, то есть INT8-движок
ведёт себя как FP16. Это **воспроизводимый научный результат** (особенность связки «трансформерная
attention-архитектура + авто-тюнер TRT»), а не ошибка кода. Поэтому 5 конфигураций RF-DETR
(`5_trt_int8_*` и `6_trt_mixed_*`) **исключены** из дипломных артефактов, и общее число валидных
конфигураций равно **35** (4 модели × 10 стадий − 5). RF-DETR проходит стадии 1–4 (FP32, ONNX, TF32,
FP16, BF16). Диагностику доли точностей по слоям даёт `analyze_engine_precision`
([`tensorrt_engine.py`](../src/benchmark/engines/tensorrt_engine.py)).

## Планируемые модели: D-FINE и DEIMv2

D-FINE и DEIMv2 уже учтены как DETR-семейство в [`utils/macs.py`](../src/benchmark/utils/macs.py) и
упомянуты в контракте `ModelAdapter`, но **адаптеры ещё не реализованы** — это фаза 10 роадмапа.
После их добавления корпус достигнет 6 моделей. См. [.planning/ROADMAP.md](../.planning/ROADMAP.md).

---

## Как добавить новую модель

1. **Создайте адаптер** `src/benchmark/models/<name>_adapter.py`, реализующий протокол
   [`ModelAdapter`](../src/benchmark/engines/pytorch_engine.py):
   - `input_size` — разрешение модели `(H, W)`;
   - `load(weights_path, device)` — загрузка весов, `eval()`, перенос на устройство, возврат `nn.Module`;
   - `preprocess(sample, device=None)` — *опционально*; реализуйте, если нужен особый препроцессинг
     (как letterbox у YOLO), иначе движок применит общий stretch-resize;
   - `infer(model, inputs)` — прямой проход;
   - `parse_outputs(raw_outputs, original_size, input_size, score_threshold)` → `Detection`
     с боксами `x1y1x2y2` в пикселях оригинала и метками в **COCO-91**.
2. **Зарегистрируйте модель** в `MODEL_REGISTRY` ([`cli.py`](../src/benchmark/cli.py)): путь к весам,
   путь к ONNX, семейство (`detr` / `yolo` / `rfdetr` — влияет на подсчёт MACs).
3. **Добавьте ветку** в `_get_adapter` ([`cli.py`](../src/benchmark/cli.py)) для создания адаптера
   по имени.
4. **(Опционально) семейство MACs:** если архитектура новая, добавьте её в `_DETR_FAMILY` или
   `_YOLO_FAMILY` в [`utils/macs.py`](../src/benchmark/utils/macs.py).
5. **Экспорт ONNX:** добавьте скрипт по образцу
   [`scripts/export_rtdetr_onnx.py`](../scripts/export_rtdetr_onnx.py) или используйте
   `export_and_simplify` / `export_yolo_to_onnx`.

После этого модель доступна во всех стадиях: `uv run benchmark run --model <name> --all-stages`.

## См. также

- [pipeline.md](pipeline.md) — стадии, через которые проходит модель.
- [architecture.md](architecture.md) — как движок взаимодействует с адаптером.
- [getting-started.md](getting-started.md) — загрузка весов и экспорт ONNX.
