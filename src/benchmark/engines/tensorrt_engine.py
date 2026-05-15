"""TensorRT inference engine for stages 3-5 benchmarking.

Supports TF32, FP16, BF16, and INT8 precision modes via a single class with
lazy engine building and serialization caching.  INT8 builds require a
calibrator (MinMax, Entropy, or Percentile) and a calibration dataloader.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from PIL import Image

from benchmark.engines.base import MEASURE_RUNS, WARMUP_RUNS, BaseEngine, Detection
from benchmark.engines.pytorch_engine import ModelAdapter
from benchmark.utils.logger import BenchmarkResult

try:
    import tensorrt as trt  # type: ignore[import-untyped]
except ImportError:
    trt = None  # type: ignore[assignment]  # TRT optional — raise only when used

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.data.coco_loader import COCODataLoader, COCOSample

logger = logging.getLogger(__name__)

# Minimum number of output tensors expected from RT-DETR TRT model
_MIN_TRT_OUTPUTS: int = 2

_INPUT_SIZE: tuple[int, int] = (640, 640)


class _BF16UnsupportedError(Exception):
    """Raised when BF16 build is attempted on unsupported hardware."""


class TensorRTEngine(BaseEngine):
    """TensorRT inference engine with lazy build and precision selection.

    Builds TRT engines from ONNX models in TF32, FP16, BF16, or INT8 precision.
    Engines are cached to disk and reused on subsequent runs unless
    ``force_rebuild=True`` is set.  INT8 builds also cache the calibration table
    to ``{engine_dir}/{model_token}_int8_{calibrator_method}.cache``.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. "rt-detr").
    precision : Literal['tf32', 'fp16', 'bf16', 'int8']
        TensorRT precision mode.
    engine_dir : Path
        Directory to cache serialized .engine files.
    adapter : ModelAdapter
        Model-specific adapter for loading and output parsing.
    force_rebuild : bool
        Force engine rebuild even if cached .engine file exists.
        For INT8, also deletes the calibration cache so calibration reruns.
    calibrator_method : Literal['minmax', 'entropy', 'percentile'] | None
        Required when ``precision='int8'``; ignored otherwise.
    score_threshold : float
        Minimum detection confidence for postprocessing.
    """

    def __init__(
        self,
        model_name: str,
        precision: Literal["tf32", "fp16", "bf16", "int8"],
        engine_dir: Path,
        adapter: ModelAdapter,
        force_rebuild: bool = False,
        calibrator_method: Literal["minmax", "entropy", "percentile"] | None = None,
        score_threshold: float = 0.001,
        mixed_strategy: Literal["a", "b"] | None = None,
    ) -> None:
        super().__init__(model_name, engine_type="tensorrt", precision=precision)
        self._adapter = adapter
        self._engine_dir = engine_dir
        self._force_rebuild = force_rebuild
        self._calibrator_method: Literal["minmax", "entropy", "percentile"] | None = (
            calibrator_method
        )
        self._score_threshold = score_threshold
        self._mixed_strategy: Literal["a", "b"] | None = mixed_strategy

        # T-07-03: sanitize model_name to alphanumeric/underscore only before using in
        # filenames, preventing path separator injection into engine_dir.
        # Dashes are replaced so rt-detr -> rt_detr (safe filesystem token).
        model_token: str = re.sub(r"[^A-Za-z0-9_]", "_", self.model_name)

        if precision == "int8":
            if calibrator_method is None:
                msg = "calibrator_method is required when precision='int8'"
                raise ValueError(msg)
            if mixed_strategy is not None:
                self._engine_path = (
                    engine_dir / f"{model_token}_mixed_{mixed_strategy}_{calibrator_method}.engine"
                )
            else:
                self._engine_path = engine_dir / f"{model_token}_int8_{calibrator_method}.engine"
            # CR-03: cache file path is namespaced per calibrator method
            # (minmax/entropy/percentile). TRT cache tables produced by
            # IInt8LegacyCalibrator (Percentile) are NOT interchangeable with
            # IInt8EntropyCalibrator2 (Entropy/MinMax); the per-method file
            # name guarantees cache isolation across algorithms. Stage 6
            # mixed-precision rebuilds share the Stage 5 cache by design
            # (D-07/D-08) — the engine filename differs by mixed_strategy
            # while the cache filename omits mixed_strategy so a/b can reuse
            # the same calibration table.
            self._cache_path: Path | None = (
                engine_dir / f"{model_token}_int8_{calibrator_method}.cache"
            )
        else:
            self._engine_path = engine_dir / f"{model_token}_{precision}.engine"
            self._cache_path = None

        self._calibration_dataloader: COCODataLoader | None = None
        self._runtime: object = None
        self._engine: object = None
        self._context: object = None
        self._stream: torch.cuda.Stream | None = None
        self._skipped_reason: str = ""
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._output_shapes: list[tuple[int, ...]] = []

    def load_model(
        self,
        weights_path: Path,
        calibration_dataloader: COCODataLoader | None = None,
    ) -> None:
        """Load or build TRT engine from ONNX model.

        Parameters
        ----------
        weights_path : Path
            Path to the .onnx model file.
        calibration_dataloader : COCODataLoader | None
            Required when ``precision='int8'`` and no cached engine exists.
            Ignored for non-INT8 precisions.
        """
        self._calibration_dataloader = calibration_dataloader
        self._engine_dir.mkdir(parents=True, exist_ok=True)

        # Force rebuild: delete existing engine AND calibration cache (INT8 only)
        if self._force_rebuild:
            if self._engine_path.exists():
                self._engine_path.unlink()
                logger.info("Deleted cached engine: %s", self._engine_path)
            if self._cache_path is not None and self._cache_path.exists():
                self._cache_path.unlink()
                logger.info("Deleted calibration cache: %s", self._cache_path)

        # INT8 build guard: calibration data required when engine must be built
        if (
            self.precision == "int8"
            and not self._engine_path.exists()
            and calibration_dataloader is None
        ):
            msg = "calibration_dataloader is required to build an INT8 engine"
            raise ValueError(msg)

        if not self._force_rebuild and self._engine_path.exists():
            logger.info("Loading cached TRT engine: %s", self._engine_path)
            self._load_engine()
            return

        logger.info(
            "Building TRT %s engine (this may take several minutes)...",
            self.precision,
        )

        if self.precision == "bf16":
            try:
                self._build_engine(weights_path)
            except _BF16UnsupportedError as exc:
                self._skipped_reason = str(exc)
                logger.warning("BF16 build skipped: %s", exc)
                return
            except Exception as exc:
                self._skipped_reason = f"BF16 build failed: {exc}"
                logger.warning("BF16 build failed, stage will be skipped: %s", exc)
                return
        else:
            self._build_engine(weights_path)

        self._load_engine()

    def _build_engine(self, onnx_path: Path) -> None:
        """Build and serialize a TRT engine from ONNX.

        Parameters
        ----------
        onnx_path : Path
            Path to the source .onnx model file.

        Raises
        ------
        RuntimeError
            If TensorRT is not installed.
        _BF16UnsupportedError
            If BF16 is requested on unsupported hardware.
        """
        if trt is None:
            msg = "TensorRT not installed. Install with: uv sync --group tensorrt"
            raise RuntimeError(msg)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        onnx_data = onnx_path.read_bytes()
        if not parser.parse(onnx_data):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            msg = f"ONNX parse failed: {errors}"
            raise RuntimeError(msg)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if self.precision == "tf32":
            config.set_flag(trt.BuilderFlag.TF32)
        elif self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "bf16":
            # BF16 hardware check via TRT Builder API (TRT-native Ampere indicator).
            # builder.platform_has_tf32 is present in TRT 10.16.1.11 and indicates
            # Ampere (sm_80+) support — used as the BF16 capability gate per D-07
            # and CLAUDE.md BF16 Verification rule.
            # trt.BuilderFlag.BF16 exists in TRT 10.x (introduced TRT 8.6+).
            if not builder.platform_has_tf32:
                msg = "BF16 not supported: platform_has_tf32=False (Ampere sm_80+ required)"
                raise _BF16UnsupportedError(msg)
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.precision == "int8":
            self._apply_int8_config(builder, network, config)
            if self._mixed_strategy:
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                from benchmark.engines.mixed_precision import apply_strategy_a, apply_strategy_b

                if self._mixed_strategy == "a":
                    count = apply_strategy_a(network)
                elif self._mixed_strategy == "b":
                    count = apply_strategy_b(network)
                logger.info(
                    "Strategy %s: %d layers set to FP16", self._mixed_strategy.upper(), count
                )

        # Inference optimization profile (batch=1) — added for all precisions.
        # ONNX was exported with dynamic_axes={0: "batch"}, so TRT
        # needs explicit min/opt/max shapes. Benchmark uses batch=1.
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            inp_shape = inp.shape  # e.g. (-1, 3, 640, 640)
            # Replace dynamic dims (-1) with 1 for batch=1 inference
            fixed = tuple(1 if d == -1 else d for d in inp_shape)
            profile.set_shape(inp.name, min=fixed, opt=fixed, max=fixed)
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            msg = f"TRT engine build failed for precision={self.precision}"
            raise RuntimeError(msg)

        self._engine_path.write_bytes(serialized)
        logger.info("TRT %s engine built and saved: %s", self.precision, self._engine_path)

    def _apply_int8_config(
        self,
        builder: object,
        network: object,
        config: object,
    ) -> None:
        """Set INT8 flag, attach calibrator, and add calibration profile (batch=8).

        Parameters
        ----------
        builder : trt.Builder
            TRT builder instance.
        network : trt.INetworkDefinition
            Parsed network.
        config : trt.IBuilderConfig
            Builder config to mutate.

        Raises
        ------
        RuntimeError
            If any required INT8 attribute is unset (internal guard).
        """
        # Local import avoids circular import at module level (TRT optional pattern).
        from benchmark.engines.int8_calibrators import _make_calibrator  # noqa: PLC0415

        if self._calibrator_method is None:
            msg = "calibrator_method is None in _apply_int8_config — internal error"
            raise RuntimeError(msg)
        if self._calibration_dataloader is None:
            msg = "calibration_dataloader is None in _apply_int8_config — internal error"
            raise RuntimeError(msg)
        if self._cache_path is None:
            msg = "cache_path is None in _apply_int8_config — internal error"
            raise RuntimeError(msg)

        calibrator = _make_calibrator(
            self._calibrator_method,
            self._calibration_dataloader,
            self._cache_path,
            adapter=self._adapter,
        )

        # Включение FP16 как Fallback для INT8.
        # Позволяет не поддерживаемым в INT8 слоям (LayerNorm, Softmax)
        # аппаратно ускоряться в FP16, а не скатываться в крайне медленный FP32.
        config.set_flag(trt.BuilderFlag.INT8)  # type: ignore[union-attr]
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[union-attr]
        config.int8_calibrator = calibrator  # type: ignore[union-attr]

        # CALIBRATE_BEFORE_FUSION: required for IInt8LegacyCalibrator (Percentile).
        # Without this flag, TRT calibrates after fusing Conv+SiLU into a single
        # node.  That fused pattern has no INT8 kernel on sm_86 (RTX 30xx), so TRT
        # raises Error Code 10 ("no implementation for node").
        # Pre-fusion calibration gives TRT per-op scale data so it can unfuse and
        # fall back the activation portion to FP16 instead of failing outright.
        # MinMax/Entropy calibrators (IInt8MinMaxCalibrator / IInt8EntropyCalibrator2)
        # handle the fused path correctly via their native TRT integration, so this
        # flag is only set for the legacy percentile path.
        if self._calibrator_method == "percentile":
            config.set_quantization_flag(  # type: ignore[union-attr]
                trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION
            )

        cal_profile = builder.create_optimization_profile()  # type: ignore[union-attr]
        for j in range(network.num_inputs):  # type: ignore[union-attr]
            inp_c = network.get_input(j)  # type: ignore[union-attr]
            cal_shape = tuple(8 if d == -1 else d for d in inp_c.shape)
            cal_profile.set_shape(inp_c.name, min=cal_shape, opt=cal_shape, max=cal_shape)
        config.set_calibration_profile(cal_profile)  # type: ignore[union-attr]

    def _load_engine(self) -> None:
        """Deserialize a cached TRT engine from disk.

        Raises
        ------
        RuntimeError
            If TensorRT is not installed or engine file is invalid.
        """
        if trt is None:
            msg = "TensorRT not installed. Install with: uv sync --group tensorrt"
            raise RuntimeError(msg)

        data = self._engine_path.read_bytes()
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(data)
        self._context = self._engine.create_execution_context()

        self._stream = torch.cuda.Stream()

        # Reset metadata before re-population — prevents double-append on retry/reload (CR-01).
        self._output_names = []
        self._output_shapes = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_names.append(name)
                self._output_shapes.append(tuple(self._engine.get_tensor_shape(name)))

        logger.info(
            "TRT engine loaded: %s (%.1f MB)",
            self._engine_path.name,
            self.model_size_mb,
        )

        try:
            analyze_engine_precision(self._engine_path)
        except Exception as e:
            logger.warning("Failed to analyze engine precision: %s", e)

    def preprocess(self, sample: COCOSample) -> np.ndarray:
        """Resize image and convert to model input tensor using adapter.

        If the adapter exposes a ``preprocess`` method (e.g. YOLO letterbox),
        delegate to it and return the tensor as a CPU numpy array. Otherwise
        fall back to the generic stretch-resize used by RT-DETR.

        Returns (1, 3, H, W) float32 numpy array.
        """
        adapter_pre = getattr(self._adapter, "preprocess", None)
        if callable(adapter_pre):
            tensor = adapter_pre(sample, device=None)
            return tensor.cpu().numpy().astype(np.float32)

        h, w = self._adapter.input_size
        img = Image.fromarray(sample.image).resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
        arr = arr.transpose(2, 0, 1)  # CHW
        return arr[np.newaxis, ...]  # (1, 3, H, W)

    def infer(self, inputs: object) -> object:
        """Run TRT inference using torch CUDA tensors for memory management.

        Parameters
        ----------
        inputs : object
            Expected: np.ndarray of shape (1, 3, H, W) float32.

        Returns
        -------
        list[np.ndarray]
            List of output tensors from the engine.
        """
        if self._context is None or self._stream is None:
            if self._skipped_reason:
                return []
            msg = "Engine not loaded. Call load_model() first."
            raise RuntimeError(msg)

        inputs_np = np.ascontiguousarray(inputs, dtype=np.float32)  # type: ignore[arg-type]

        # Allocate GPU memory via torch tensors
        input_gpu = torch.as_tensor(inputs_np, device="cuda")
        self._context.set_tensor_address(self._input_name, input_gpu.data_ptr())

        # Allocate output buffers
        output_tensors: list[torch.Tensor] = []
        for name, shape in zip(self._output_names, self._output_shapes, strict=True):
            out_gpu = torch.empty(shape, dtype=torch.float32, device="cuda")
            self._context.set_tensor_address(name, out_gpu.data_ptr())
            output_tensors.append(out_gpu)

        # Выполнение в изолированном потоке без блокировки GPU-дефолта
        self._context.execute_async_v3(self._stream.cuda_stream)

        # Строгая синхронизация только выделенного потока перед копированием
        self._stream.synchronize()

        # Copy outputs to CPU
        return [t.cpu().numpy() for t in output_tensors]

    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Delegate to adapter for model-specific output parsing.

        Parameters
        ----------
        raw_outputs : object
            List of numpy arrays from inference.
        sample : COCOSample
            Original image metadata.
        """
        return self._adapter.parse_outputs(
            raw_outputs,
            original_size=sample.original_size,
            input_size=self._adapter.input_size,
            score_threshold=self._score_threshold,
        )
    @property
    def model_size_mb(self) -> float:
        """TRT engine file size in MB."""
        if not self._engine_path.exists():
            return 0.0
        return self._engine_path.stat().st_size / (1024 * 1024)

    def run_full_benchmark(
        self,
        dataloader: COCODataLoader,
        stage: str = "3_trt_tf32",
        baseline_map_50_95: float = 0.0,
        macs: float | None = None,
        flops: float | None = None,
    ) -> BenchmarkResult:
        """Run complete benchmark, handling skipped engines gracefully.

        If the engine was skipped (e.g. BF16 unsupported), returns a
        BenchmarkResult with NaN metrics and skipped_reason populated.
        Otherwise delegates to BaseEngine.run_full_benchmark().

        Parameters
        ----------
        dataloader : COCODataLoader
            Data loader with COCO val2017 images.
        stage : str
            Stage identifier string (D-04).
        baseline_map_50_95 : float
            FP32 baseline mAP for accuracy drop calculation.
        macs : float | None
            Pre-computed MACs.
        flops : float | None
            Pre-computed FLOPs.
        """
        if self._skipped_reason:
            nan = float("nan")
            return BenchmarkResult(
                model_name=self.model_name,
                stage=stage,
                engine_type="tensorrt",
                precision=self.precision,
                latency_preprocess_ms=nan,
                latency_inference_ms=nan,
                latency_postprocess_ms=nan,
                latency_total_ms=nan,
                throughput_fps=nan,
                jitter_ms=nan,
                map_50_95=nan,
                map_50=nan,
                map_75=nan,
                map_small=nan,
                map_medium=nan,
                map_large=nan,
                ar_1=nan,
                ar_10=nan,
                ar_100=nan,
                ar_small=nan,
                ar_medium=nan,
                ar_large=nan,
                accuracy_drop_pct=nan,
                model_size_mb=0.0,
                vram_peak_mb=0.0,
                macs=macs,
                flops=flops,
                warmup_runs=WARMUP_RUNS,
                measure_runs=MEASURE_RUNS,
                skipped_reason=self._skipped_reason,
            )

        return super().run_full_benchmark(
            dataloader,
            stage=stage,
            baseline_map_50_95=baseline_map_50_95,
            macs=macs,
            flops=flops,
        )


def analyze_engine_precision(engine_path: Path) -> dict[str, int | float]:
    """
    Анализирует скомпилированный TRT engine и возвращает послойную статистику
    аппаратного квантования (INT8 vs FP16 vs FP32).

    Parameters
    ----------
    engine_path : Path
        Путь к сериализованному .engine файлу.

    Returns
    -------
    dict[str, int | float]
        Словарь с абсолютными счетчиками слоев и процентной долей INT8.

    Raises
    ------
    RuntimeError
        Если движок не удалось десериализовать, либо если TensorRT не установлен.
    """
    if trt is None:
        msg = "TensorRT not installed. Install with: uv sync --group tensorrt"
        raise RuntimeError(msg)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)

    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if not engine:
        raise RuntimeError(f"Не удалось десериализовать TRT engine: {engine_path}")

    inspector = engine.create_engine_inspector()
    # Выгружаем профиль графа в машиночитаемом формате
    engine_json_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    engine_data = json.loads(engine_json_str)

    layers = engine_data.get("Layers", [])
    total_layers = len(layers)

    if total_layers == 0:
        logger.warning("Engine Inspector вернул пустой список слоев.")
        return {"total_layers": 0, "int8_ratio_percent": 0.0}

    stats: dict[str, int] = {"INT8": 0, "FP16": 0, "FP32": 0, "OTHER": 0, "UNKNOWN": 0}

    for layer in layers:
        if isinstance(layer, dict):
            # Пытаемся получить явное поле Precision (если есть)
            precision = layer.get("Precision", None)
            if not precision:
                # Если явного нет, смотрим на тип данных первого выхода
                outputs = layer.get("Outputs", [])
                if outputs and isinstance(outputs, list):
                    datatype = outputs[0].get("Format/Datatype", "OTHER")
                    if datatype == "Int8":
                        precision = "INT8"
                    elif datatype == "Half":
                        precision = "FP16"
                    elif datatype == "Float":
                        precision = "FP32"
                    else:
                        precision = "OTHER"
                else:
                    precision = "OTHER"
        else:
            # Если профилирование не DETAILED, слой — это просто строка (имя)
            precision = "UNKNOWN"

        if precision in stats:
            stats[precision] += 1
        else:
            stats["OTHER"] += 1

    # Считаем процент INT8 только среди известных слоев, либо среди всех
    known_layers = total_layers - stats["UNKNOWN"]
    if known_layers > 0:
        int8_ratio = (stats["INT8"] / known_layers) * 100
    else:
        int8_ratio = 0.0

    metrics: dict[str, int | float] = {
        "total_layers": total_layers,
        "int8_layers": stats["INT8"],
        "fp16_layers": stats["FP16"],
        "fp32_layers": stats["FP32"],
        "other_layers": stats["OTHER"],
        "unknown_layers": stats["UNKNOWN"],
        "int8_ratio_percent": round(int8_ratio, 2),
    }

    logger.info(
        "Engine Precision Profile | Total: %d | INT8: %d (%.2f%%) | FP16: %d | FP32: %d | Other: %d | Unknown: %d",
        metrics["total_layers"],
        metrics["int8_layers"],
        metrics["int8_ratio_percent"],
        metrics["fp16_layers"],
        metrics["fp32_layers"],
        metrics["other_layers"],
        metrics["unknown_layers"],
    )

    return metrics
