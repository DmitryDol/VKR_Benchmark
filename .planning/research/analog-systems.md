# Системы-аналоги: бенчмаркинг и оптимизация инференса детекторов

> Подготовлено для слайда защиты ВКР. Дата: 2026-05-27.
> Тема: «Аппаратная оптимизация и бенчмаркинг инференса трансформер-based детекторов на NVIDIA RTX 3070».

---

## 1. Системы-аналоги

Ниже — пять наиболее релевантных систем/работ, делающих похожее: бенчмаркинг и/или оптимизация инференса детекторов на TensorRT/INT8.

### 1.1. MLPerf Inference (MLCommons), v5.0 / v5.1 — 2025

- **Что делают:** индустриальный стандарт бенчмаркинга инференса. В разделе Vision/Edge — RetinaNet (Offline/Server) и Automotive 3D Object Detection (PointPainting).
- **Метрики:** Throughput (samples/s), Latency (ms, server scenario), Accuracy threshold (~99% от FP32 mAP).
- **Железо:** от RTX 4000 Ada / RTX Pro 6000 Blackwell до 8× H200, GB300, MI355X.
- **Источник:** [MLCommons MLPerf Inference v5.0 Results](https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/), [HPCwire — v5.1, Sep 2025](https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/).

### 1.2. NVIDIA TensorRT Model Optimizer (TRT-LLM / Vision)

- **Что делают:** open-source toolkit от NVIDIA для PTQ/QAT, INT8 / FP8 / INT4 / FP4. Поддержка калибровок MinMax / Entropy / SmoothQuant, экспорт в ONNX/TRT.
- **Фокус:** в первую очередь LLM, для CV-моделей даёт калибровщики + примеры (samples/python/efficientdet, samples/python/yolo).
- **Источник:** [NVIDIA Technical Blog — TensorRT Model Optimizer](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/), [docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-transformers.html).

### 1.3. Q-DETR / AQ-DETR — академические работы по квантованию DETR

- **Q-DETR (CVPR 2023):** 4-битное PTQ для DETR-R50, достигает 76.9% AP50 (gap 6.4% против FP32). Предлагает Distribution Rectification Distillation.
- **AQ-DETR (AAAI 2024):** Low-bit квантование с auxiliary queries для уменьшения деградации в attention.
- **Источник:** [Q-DETR (arXiv:2304.00253)](https://arxiv.org/pdf/2304.00253), [AQ-DETR](https://ojs.aaai.org/index.php/AAAI/article/download/29487/30803), [Efficient Integer Quantization for Compressed DETR (PMC12025429)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025429/).

### 1.4. «Quantization Robustness to Input Degradations for Object Detection» (arXiv:2508.19600, 2025)

- **Что делают:** Static INT8 PTQ через TensorRT на семействе детекторов (YOLO, RT-DETR) на COCO + 7 degradation conditions. Сравнение калибровок, degradation-aware calibration strategy.
- **Результаты:** Static INT8 TRT даёт **1.5–3.3× speedup** при **3–7% mAP50:95 drop** на чистых данных.
- **Источник:** [arXiv:2508.19600](https://arxiv.org/pdf/2508.19600).

### 1.5. Roboflow RF-DETR official benchmark + Ultralytics YOLO benchmarks

- **RF-DETR (Roboflow, ICLR 2026):** официальный TRT FP16 benchmark на T4. RF-DETR-L: **56.5 AP@50:95 за 6.8 ms (FP16, T4, BS=1)**, RF-DETR-Nano: 48.4 AP за 2.32 ms (~431 FPS).
- **Ultralytics YOLO11/YOLO26:** официальные docs приводят FP16/INT8 экспорт + сравнение форматов. INT8 калибровка даёт +20–40% к скорости при «небольшом» mAP drop.
- **Источник:** [RF-DETR (Roboflow Docs)](https://rfdetr.roboflow.com/develop/), [RF-DETR NAS (arXiv:2511.09554)](https://arxiv.org/html/2511.09554v1), [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt), [Peter Makhalov — YOLO TensorRT comparison](https://medium.com/@peter.makhalov/comparing-performance-of-yolo-family-object-detectors-for-tensorrt-implementations-69e7e8e42c69).

---

## 2. Сравнительные результаты

Сводная таблица: что измеряет каждая система, на каком железе, какой получает выигрыш и просадку точности.

| # | Система / работа | Модели | Железо | Stages | Speedup (vs FP32) | mAP drop |
|---|------------------|--------|--------|--------|-------------------|----------|
| 1 | **MLPerf Inference v5.0/v5.1** | RetinaNet, 3D-OD PointPainting | H100/H200, RTX Pro 6000, MI355X | FP16/INT8/FP8 | n/a (абсолютные FPS) | ≤1% (gate) |
| 2 | **TRT Model Optimizer (samples)** | EfficientDet, YOLO | A100, RTX 4090, Jetson | FP16, INT8 (Max/Entropy), FP8, INT4 | 2–4× (FP16), 3–5× (INT8) | 1–3% (INT8 typical) |
| 3 | **Q-DETR** (CVPR 2023) | DETR-R50 | V100 (training) | 4-bit PTQ + DRD | n/a (теоретический) | **−6.4% AP50** (4-bit) |
| 4 | **Quant. Robustness…** (arXiv 2508.19600) | YOLO + RT-DETR семейство | RTX-class GPU | Static INT8 TRT | **1.5–3.3×** | **3–7% mAP50:95** |
| 5 | **RF-DETR (Roboflow)** | RF-DETR-N/S/M/L/2XL | NVIDIA T4 | TRT FP16 (BS=1) | n/a (FP16 only) | негласно ~0% |
| 6 | **Ultralytics YOLO11/26 docs** | YOLO11, YOLO26 | RTX 3070+ / Jetson | ONNX, TRT FP16, TRT INT8 | FP16: 1.3–3.4×, INT8: +20–40% к FP16 | INT8: «minor» |
| 7 | **Peter Makhalov — YOLO TRT** | YOLOv5/v8/v10/v11 | RTX-class + Jetson | TRT FP16, INT8 PTQ | FP16: **1.26–2.40×**, INT8: 1.5–3.3× | INT8: 3–7% mAP50:95 |
| 8 | **DETR на Jetson Thor (RF-DETR XLarge Seg)** | RT-DETR / RF-DETR | Jetson Thor | FP16, FP8, INT8 | FP16→INT8: только **+5–10%** | n/a |

**Ключевые наблюдения для защиты:**

1. У DETR-семейства выигрыш от **INT8 поверх FP16 скромный (5–10%)** — узким местом становятся attention/LayerNorm, а не GEMM. Это обосновывает необходимость mixed precision и sensitivity analysis в ВКР.
2. Типичный коридор просадки mAP при PTQ INT8 на детекторах — **3–7% mAP50:95** (статья arXiv:2508.19600). ВКР целится в этот же коридор и проверяет его на 6 разных архитектурах одновременно.
3. Ни один из аналогов **не делает одновременно**: per-class AP + confusion matrix + jitter + sensitivity analysis для transformer-детекторов в одном репортинге.

---

## 3. Отличия и преимущества разрабатываемой системы

Что делает ВКР, чего нет (или не сделано целостно) у аналогов:

1. **Покрытие зоопарка SOTA-детекторов в одной системе.** Шесть моделей одной кодовой базой: RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26. MLPerf и Roboflow тестируют 1–2 модели; академические работы (Q-DETR, AQ-DETR) — только одну.

2. **Шесть стадий оптимизации в едином пайплайне.** PyTorch FP32 → ONNX (+ simplifier) → TRT TF32 → TRT FP16/BF16 → TRT INT8 (3 калибровки) → Mixed Precision INT8+FP16 (3 стратегии). Каждый шаг логируется отдельно. У TRT Model Optimizer и Ultralytics — только финальная стадия.

3. **Три независимых INT8-калибратора (MinMax / Entropy / Percentile) с прямым сравнением mAP/Latency.** В аналогах обычно используется одна калибровка (Entropy в TRT Model Optimizer, MinMax в Ultralytics). ВКР даёт научное сравнение всех трёх на одинаковых моделях.

4. **Три mixed-precision стратегии + опциональный Sensitivity Analysis.**
   - Strategy A: первый/последний слой FP16, остальное INT8.
   - Strategy B: Softmax/LayerNorm в FP16 (важно именно для DETR).
   - Strategy C: программный поиск N% самых чувствительных слоёв (Hessian/Information-Flow-driven).
   Это закрывает разрыв «INT8→FP16 даёт 5–10%» из RF-DETR/Jetson Thor: stratey B/C целенаправленно атакует attention-блоки.

5. **Расширенный набор метрик с научной строгостью.** Latency (pre+infer+post), Throughput, **Jitter (std)**, mAP50, mAP50:95, **per-class AP**, **confusion matrix**, IoU, Model Size, **VRAM (peak)**, MACs/FLOPs — всё в CSV+JSON по каждой стадии. MLPerf даёт только Throughput + один accuracy gate; Ultralytics docs не приводят jitter и per-class AP; Q-DETR/AQ-DETR — только mAP. ВКР даёт publication-ready отчёты без потери промежуточных результатов.

6. **Бюджетное железо и фиксированные условия.** Single RTX 3070 (Ampere, 8GB), batch=1, TF32 принудительно выключен для baseline, 50 warmup + 1000 measure, TRT workspace ровно 2 GB. Воспроизводимость выше, чем у MLPerf (где железо разное у каждого вендора).

---

## Ключевые источники (для слайда «Литература»)

- [MLCommons MLPerf Inference v5.0 (Apr 2025)](https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/)
- [MLPerf Inference v5.1 (HPCwire, Sep 2025)](https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/)
- [NVIDIA TensorRT Model Optimizer (blog)](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- [TensorRT — Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [Q-DETR: An Efficient Low-Bit Quantized Detection Transformer (arXiv:2304.00253)](https://arxiv.org/pdf/2304.00253)
- [AQ-DETR (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/download/29487/30803)
- [Quantization Robustness to Input Degradations for OD (arXiv:2508.19600)](https://arxiv.org/pdf/2508.19600)
- [RF-DETR (Roboflow Docs)](https://rfdetr.roboflow.com/develop/) и [RF-DETR NAS (arXiv:2511.09554)](https://arxiv.org/html/2511.09554v1)
- [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt)
- [D-FINE (arXiv:2410.13842)](https://arxiv.org/html/2410.13842v1)
- [Mix-QViT — sensitivity-driven mixed precision (arXiv:2501.06357)](https://arxiv.org/html/2501.06357v1)
- [Peter Makhalov — YOLO TRT benchmarks (Medium)](https://medium.com/@peter.makhalov/comparing-performance-of-yolo-family-object-detectors-for-tensorrt-implementations-69e7e8e42c69)

---

## Системы-аналоги, специализированные на Object Detection

Раздел добавлен в ответ на замечание: ранее найденные аналоги (MLPerf, TRT Model Optimizer) воспринимаются как LLM-ориентированные. Ниже — системы и фреймворки, делающие бенчмаркинг и оптимизацию **исключительно для детекторов** (YOLO, DETR, RT-DETR, RF-DETR, D-FINE, DEIMv2 и пр.).

### OD-1. Ultralytics `model.benchmark()` (YOLO11 / YOLO26)

- **Что делают:** официальный API Ultralytics `from ultralytics.utils.benchmarks import benchmark` и режим `model.benchmark(...)`. Один вызов прогоняет YOLO-чекпойнт через **11 форматов экспорта** (PyTorch, TorchScript, ONNX, OpenVINO, TensorRT FP16/INT8, CoreML, TF SavedModel, TFLite, PaddlePaddle, NCNN) и печатает таблицу `format | size_mb | mAP50-95 | inference_ms | FPS`.
- **Модели:** вся линейка YOLO11/YOLO26 (n/s/m/l/x), плюс YOLOv8/v9/v10/v12, RT-DETR, SAM, YOLO-World.
- **Железо:** официально документированы прогоны на Jetson Orin Nano/NX/AGX, RTX 3080 Laptop, RTX 4090, T4. RTX 3070 не приведён в docs, но скрипт `benchmark()` тривиально запускается на нём.
- **Precision:** FP32 / FP16 (`half=True`) / INT8 (`int8=True`, MinMax-калибровка на 100 изображениях COCO).
- **Числа:** на RTX 3080 Laptop YOLOv8n PyTorch ≈ 15–20 ms/frame → TRT FP16 ≈ 5–8 ms/frame (×2–4 speedup); YOLOv8s INT8 = 3.2 ms (313 FPS). FP16 speedup на разных GPU: **+26 % … +240 %** к FP32, mAP drop при FP16 — пренебрежимо малый.
- **Источники:** [Ultralytics benchmark docs](https://docs.ultralytics.com/modes/benchmark), [TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt), [benchmarks.py reference](https://docs.ultralytics.com/reference/utils/benchmarks), [GitHub `ultralytics/docs/modes/benchmark.md`](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/benchmark.md), [Ultralytics blog «How to benchmark YOLO11»](https://www.ultralytics.com/blog/how-to-benchmark-ultralytics-yolo-models-like-yolo11).

### OD-2. OpenMMLab MMDeploy (для MMDetection)

- **Что делают:** официальный deployment-фреймворк OpenMMLab. Конвейер `PyTorch → ONNX → backend (TensorRT / ONNX Runtime / OpenVINO / ncnn / PPLNN / CoreML)` с автоматическим прогоном `tools/profiler.py` и `tools/test.py` для замера Latency, FPS, mAP на COCO.
- **Модели:** RTMDet (SOTA real-time), YOLOX, YOLOv6/v7/v8, Faster R-CNN, Mask R-CNN, RetinaNet, FCOS, DETR, Deformable-DETR, DINO, RT-DETR (через mmdet 3.x).
- **Precision:** FP32 / FP16 / INT8 PTQ (через TensorRT EntropyCalibrator2 + калибровочный датасет COCO).
- **Что важно для ВКР:** прямой аналог нашего пайплайна Stage 1 → Stage 5, но привязанный к экосистеме mmdet (config-driven). У нас — нативный PyTorch без mm-зависимостей, шире набор калибровок (MinMax/Entropy/Percentile) и mixed-precision стратегии.
- **Источники:** [MMDeploy docs — MMDetection deployment](https://mmdeploy.readthedocs.io/en/stable/04-supported-codebases/mmdet.html), [MMDeploy — How to convert model](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html), [GitHub open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection).

### OD-3. PaddleDetection + PaddleSlim (Baidu PP-YOLOE / RT-DETR)

- **Что делают:** Baidu-овский аналог Ultralytics+TRT. PaddleSlim предоставляет **PTQ (offline) и QAT (online)** для всех моделей PaddleDetection с экспортом в Paddle Inference / TensorRT.
- **Модели:** PP-YOLOE / PP-YOLOE+, **RT-DETR (оригинальная имплементация авторов!)**, PP-PicoDet, YOLOv5/v6/v7/v8/v10/v11, YOLOX, RTMDet.
- **Precision:** FP32 / FP16 / INT8 PTQ / INT8 QAT.
- **Числа:** PaddleSlim QAT для YOLO-семейства даёт **+30 % к скорости при near-lossless mAP** (заявлено в docs). PTQ для PP-YOLOE: typical mAP drop < 1 % на T4 и V100.
- **Зачем в обзоре:** именно PaddleDetection — родина RT-DETR; их бенчмарки на T4/V100 (TRT FP16) — золотой стандарт, на который ссылается academic community.
- **Источники:** [GitHub PaddleDetection — PaddleYOLO models](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8/docs/feature_models/PaddleYOLO_MODEL_en.md), [PaddleYOLO README](https://github.com/PaddlePaddle/PaddleYOLO/blob/release/2.5/README_en.md), [Paddle-Inference TRT integration](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/optimize/paddle_trt_en.rst), [paddleslim_detection пример](https://github.com/Sco-cai/paddleslim_detection).

### OD-4. Deci YOLO-NAS + SuperGradients (PTQ/QAT for OD)

- **Что делают:** Deci AI выпустила YOLO-NAS с **архитектурой, специально спроектированной под квантование** (quantization-aware blocks). SuperGradients включает full PTQ + QAT pipeline с экспортом в INT8 TensorRT (`trtexec --fp16 --int8`).
- **Модели:** YOLO-NAS-S / M / L (есть варианты `_int8`).
- **Числа:** mAP drop при PTQ INT8 — **0.51 / 0.65 / 0.45 пункта** для S/M/L (минимальная деградация в индустрии; типичный YOLOv5/v8 теряет 1–2 пункта).
- **Почему важно:** прямой контрпример к «INT8 = большой drop» из академических работ. Показывает, что архитектурный co-design + правильная калибровка ≈ нулевой drop.
- **Источники:** [SuperGradients — YOLO-NAS benchmarks doc](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/BenchmarkingYoloNAS.md), [QAT/PTQ guide](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/qat_ptq_yolo_nas.md), [YOLO-NAS README](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md), [AMD Quark — YOLO-NAS quantization sample](https://quark.docs.amd.com/latest/pytorch/sample_yolo_nas_quant.html).

### OD-5. NVIDIA DeepStream SDK + reference YOLO/RT-DETR apps

- **Что делают:** production-grade video-analytics SDK от NVIDIA. Включает готовые pipelines для YOLOv5/v7/v8/v11/v26 и (через `nvinferserver` + Triton) для RT-DETR / DETR. `trtexec` используется для bench, DeepStream — для end-to-end (decode + preproc + infer + tracker).
- **Модели:** все YOLO от v3 до v26 (через [DeepStream-Yolo от marcoslucianops](https://github.com/marcoslucianops/DeepStream-Yolo) и [NVIDIA-AI-IOT/yolo_deepstream](https://github.com/NVIDIA-AI-IOT/yolo_deepstream)), плюс TAO-detection nets (DetectNet_v2, EfficientDet, DINO).
- **Precision:** FP32 / FP16 / INT8 (Entropy-калибровка через `nvinfer` config).
- **Числа:** INT8 vs FP16 даёт **+40…+80 %** к throughput на Ampere/Ada-классе GPU и Jetson Orin (заявлено NVIDIA). На Jetson Orin NX YOLOv8m INT8 ≈ 30+ FPS end-to-end.
- **Зачем в обзоре:** показывает, что наш стенд (RTX 3070 + TRT 10.x) — это «единичный pipeline» из DeepStream без оркестрации Gst-пайплайна, что обоснованно для научного бенчмарка.
- **Источники:** [DeepStream SDK landing](https://developer.nvidia.com/deepstream-sdk), [DeepStream performance docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html), [Ultralytics — YOLO26 + DeepStream guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson), [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo), [NVIDIA-AI-IOT/yolo_deepstream](https://deepwiki.com/NVIDIA-AI-IOT/yolo_deepstream).

### OD-6. Roboflow RF-DETR (official benchmark) + `roboflow/supervision`

- **Что делают:** Roboflow поддерживает open-source RF-DETR (ICLR 2026, SOTA на COCO) с публичными бенчмарками в `rfdetr.roboflow.com`. Метрика «Total Latency» (preproc + infer + NMS-free postproc) на T4 + TRT10 FP16. Библиотека `supervision` даёт инструменты для side-by-side comparison YOLO vs RF-DETR (одинаковые dataset/metrics).
- **Модели:** RF-DETR-N / S / M / L / 2XL, plus упоминают сравнения с YOLO11x, YOLOv8x, YOLOv10x.
- **Числа на T4 (TRT FP16, BS=1):**
  - RF-DETR-Nano: 48.4 AP50:95 за 2.32 ms (**≈431 FPS**),
  - RF-DETR-L: **56.5 AP** за 6.8 ms (**~147 FPS**), выше чем YOLO11x (54.7 AP) при той же или меньшей latency,
  - RF-DETR-Small: +1.8 AP vs YOLO11x при -7.77 ms latency.
- **Опыт INT8 на Jetson Thor:** в issue [rf-detr#955](https://github.com/roboflow/rf-detr/issues/955) команда RF-DETR честно заявляет: **FP16 → INT8 даёт прирост только +5…+10 %**, основной выигрыш — FP32 → FP16. Это прямое эмпирическое подтверждение тезиса ВКР о слабой эффективности «сырого» INT8 на DETR-семействе.
- **Источники:** [RF-DETR benchmarks](https://rfdetr.roboflow.com/develop/learn/benchmarks/), [GitHub roboflow/rf-detr](https://github.com/roboflow/rf-detr), [RF-DETR vs YOLO11 issue #480](https://github.com/roboflow/rf-detr/issues/480), [Roboflow blog «Best OD models 2026»](https://blog.roboflow.com/best-object-detection-models/), [arXiv 2511.09554 (RF-DETR NAS)](https://arxiv.org/pdf/2511.09554), [LearnOpenCV RF-DETR review](https://learnopencv.com/rf-detr-object-detection/).

### OD-7. `kentaroy47/benchmark-FP32-FP16-INT8-with-TensorRT` (academic reference repo)

- **Что делают:** компактный open-source бенчмарк (320 ★), измеряющий **FP32 / FP16 / INT8** с TensorRT на Jetson Nano / Xavier и desktop-картах. Прозрачный код для воспроизведения цифр.
- **Модели:** YOLOv3, SSD, ResNet (классификаторы как baseline).
- **Зачем в обзоре:** академический шаблон, на который ссылаются десятки работ. ВКР делает то же самое, но для **6 transformer-детекторов + 3 калибровки + 3 mixed-precision стратегии + extended metrics**.
- **Источник:** [GitHub kentaroy47/benchmark-FP32-FP16-INT8-with-TensorRT](https://github.com/kentaroy47/benchmark-FP32-FP16-INT8-with-TensorRT).

### OD-8. D-FINE и DEIMv2 — официальные TRT FP16 бенчмарки авторов

- **D-FINE (arXiv:2410.13842):** авторы измеряют latency на **NVIDIA T4 под TensorRT FP16**. D-FINE-L: 54.0 AP @ 124 FPS, D-FINE-X: 55.8 AP @ 78 FPS. INT8-бенчмарков в paper нет — это пробел, который и закрывает ВКР.
- **DEIMv2 (arXiv:2509.20787, BrightCoding blog 2026):** новая SOTA real-time detection поверх DINOv3-backbone, той же RT-DETR-style архитектуры. Все evaluations — на T4 + TRT FP16. INT8 не рассматривается авторами.
- **RT-DETRv4 (arXiv:2510.25257, Oct 2025):** RT-DETRv4-L = 55.4 AP @ 124 FPS, RT-DETRv4-X = 57.0 AP. Тот же стенд (T4, TRT FP16). Тоже без INT8.
- **Вывод:** все три ключевые работы 2024–2026 (D-FINE, DEIMv2, RT-DETRv4) ограничиваются TRT FP16 и **не публикуют INT8 / mixed precision**. ВКР заполняет эту нишу — система тестирует Stage 4 (FP16/BF16) + Stage 5 (INT8 ×3 калибровки) + Stage 6 (Mixed) для всего DETR-семейства.
- **Источники:** [D-FINE arXiv:2410.13842](https://arxiv.org/html/2410.13842v1), [DEIMv2 arXiv:2509.20787](https://arxiv.org/html/2509.20787), [GitHub Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), [RT-DETRv4 arXiv:2510.25257](https://arxiv.org/pdf/2510.25257), [DEIMv2 blog (BrightCoding)](https://www.blog.brightcoding.dev/2026/04/22/deimv2-the-revolutionary-real-time-detector-powered-by-dinov3).

### Сравнительная таблица OD-аналогов

| # | Система | Модели OD | Backend | FP16 | INT8 PTQ | INT8 QAT | Mixed Prec. | Per-class AP | Jitter | Sensitivity Analysis |
|---|---------|-----------|---------|:----:|:--------:|:--------:|:-----------:|:------------:|:------:|:--------------------:|
| OD-1 | Ultralytics `benchmark()` | YOLOv8–YOLO26, RT-DETR | TRT/ORT/OV/CoreML | + | + (MinMax) | − | − | − | − | − |
| OD-2 | MMDeploy (mmdet) | RTMDet, DETR, RT-DETR, YOLOX | TRT/ORT/OV/ncnn | + | + (Entropy) | − | − | − | − | − |
| OD-3 | PaddleDetection + PaddleSlim | PP-YOLOE, RT-DETR | Paddle/TRT | + | + | + | − | − | − | − |
| OD-4 | SuperGradients (YOLO-NAS) | YOLO-NAS | TRT | + | + | + | − | − | − | − |
| OD-5 | DeepStream SDK | YOLOv3–v26, DINO | TRT (через nvinfer) | + | + (Entropy) | − | − | − | − | − |
| OD-6 | RF-DETR / Roboflow | RF-DETR N/S/M/L/2XL | TRT | + | − (только FP8/INT8 эксп.) | − | − | − | − | − |
| OD-7 | kentaroy47 repo | YOLOv3, SSD | TRT | + | + | − | − | − | − | − |
| OD-8 | D-FINE / DEIMv2 / RT-DETRv4 papers | D-FINE, DEIMv2, RT-DETRv4 | TRT | + | − | − | − | − | − | − |
| **ВКР** | **наш стенд** | **RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26** | **PyTorch+ONNX+TRT** | **+ FP16 и BF16** | **+ (MinMax/Entropy/Percentile)** | − | **+ (A/B/C-Sensitivity)** | **+** | **+** | **+ (опц.)** |

**Что подсвечивает таблица для защиты:**

1. **Покрытие:** ни один из 8 аналогов не делает full-stack бенчмарк ≥4 разных transformer-детекторов одновременно. Ultralytics — только YOLO, RF-DETR — только RF-DETR, mmdet — конфиг-driven и без mixed-precision.
2. **Калибровки:** только PaddleSlim + наш стенд предлагают QAT/3 PTQ-метода. Из «чистых PTQ»-фреймворков три независимых калибратора (MinMax/Entropy/Percentile) с side-by-side mAP — уникальное предложение ВКР.
3. **Mixed precision:** **никто из аналогов не публикует Strategy A/B/C** (first+last FP16 / Softmax+LayerNorm FP16 / sensitivity-driven). ВКР — первая публичная реализация для DETR-семейства.
4. **Метрики:** только ВКР даёт jitter + per-class AP + confusion matrix в одном репортинге.
5. **Эмпирическое подтверждение из OD-6:** Roboflow в issue #955 сами признают, что INT8 поверх FP16 даёт всего +5…+10 % на DETR. Это и есть мотивация для Strategy B (Softmax/LayerNorm в FP16) и Strategy C (sensitivity).
