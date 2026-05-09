# Project Context: Transformer-based Object Detection Benchmarking

## Цель проекта

Создание production-ready системы аппаратной оптимизации и бенчмаркинга инференса (mAP, Latency) для нейронных сетей на основе трансформеров без обучения с нуля. Результаты предназначены для академического диплома.

## Технический стек и Инструменты

- Language: Python 3.13
- Package Manager: uv
- Linter/Formatter: ruff (strict mode)
- Deep Learning: PyTorch, ONNX, TensorRT (Python API), COCO API
- CLI: typer или argparse
- Модели для тестирования: RT-DETR, RF-DETR, D-FINE, DEIMv2, YOLO11, YOLO26.

## Целевое оборудование и Ограничения

- Target GPU: NVIDIA RTX 3070 (Ampere, sm_86, 8 GB VRAM).
- TRT Workspace Limit: Строго 2 GB (`config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)`).
- Batch Size: Строго 1 (имитация real-time).
- Warm-up: 50 прогонов перед замерами. Замеры усредняются по 1000 итерациям.

## Глобальные инженерные правила

1. Baseline Integrity: При работе с чистым PyTorch (FP32) аппаратное ускорение TF32 должно быть принудительно отключено (`torch.backends.cuda.matmul.allow_tf32 = False`).
2. Memory Profiling: Пиковое потребление видеопамяти фиксировать строго через `torch.cuda.max_memory_allocated()`. Обязательно освобождать память и очищать кэш CUDA между инициализациями разных движков.
3. Code Quality: Код должен быть строго типизирован. Логика должна быть модульной (отдельно DataLoader, отдельно Engine, отдельно Logger).
4. Data Flow: Изображения `data/val2017`, аннотации `data/annotations`.
5. BF16 Verification: Поддержка аппаратного Bfloat16 должна проверяться перед сборкой движка (`builder.platform_has_fast_native_fp16`).

## Optimization Pipeline (Core Logic)

Код должен поддерживать проведение каждой модели через следующий пайплайн экспериментов:

1. Stage 1: Baseline (PyTorch FP32). (Учитывая правило Baseline Integrity).
2. Stage 2: ONNX Export. Обязательное применение `onnx-simplifier` к вычислительному графу.
3. Stage 3: TensorRT TF32. 32-битная точность, но с передачей флага `trt.BuilderFlag.TF32` для активации тензорных ядер Ampere.
4. Stage 4: TensorRT Half Precision. Два независимых билда: классический FP16 и аппаратно поддерживаемый BF16.
5. Stage 5: TensorRT INT8 (Extreme Compression). Обязательная реализация трех модулей калибровки: 
   - MinMax Calibration
   - Entropy Calibration
   - Percentile Calibration
6. Stage 6: Mixed Precision Quantization (INT8 + FP16). Использование лучшего метода из Stage 5 с fallback-стратегиями:
   - Strategy A: Первый и последний слои сети в FP16, остальное — INT8.
   - Strategy B: Все блоки Softmax и LayerNorm в FP16, остальное — INT8.
   - Strategy C (Sensitivity Analysis): Программный поиск N% самых чувствительных к потере точности слоев и сохранение их в FP16. **Важно:** Данная стратегия должна иметь возможность полного отключения (опциональный запуск через явный CLI-флаг, например `--enable-sensitivity-analysis`), так как процесс профилирования занимает длительное время. По умолчанию эта стратегия отключена.

## Логируемые метрики

1. Latency (ms): Pre-processing + Inference + Post-processing.
2. Throughput (FPS).
3. Jitter (ms): Стандартное отклонение времени инференса.
4. mAP (mAP_50, mAP_50:95) & Accuracy Drop (%).
5. IoU.
6. Model Size (MB) & VRAM Usage (MB).
7. MACs / FLOPs.
8. Формат сохранения: Все метрики по каждому этапу пайплайна должны структурированно сохраняться в `.csv` и `.json` файлы для последующего анализа и вставки в дипломную работу.