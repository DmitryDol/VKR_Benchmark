"""TensorRT inference engine for stages 3-4 benchmarking.

Supports TF32, FP16, and BF16 precision modes via a single class with
lazy engine building and serialization caching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from PIL import Image

from benchmark.data.coco_loader import COCO_80_TO_91
from benchmark.engines.base import MEASURE_RUNS, WARMUP_RUNS, BaseEngine, Detection
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

    Builds TRT engines from ONNX models in TF32, FP16, or BF16 precision.
    Engines are cached to disk and reused on subsequent runs unless
    ``force_rebuild=True`` is set.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. "rt-detr").
    precision : Literal['tf32', 'fp16', 'bf16']
        TensorRT precision mode.
    engine_dir : Path
        Directory to cache serialized .engine files.
    force_rebuild : bool
        Force engine rebuild even if cached .engine file exists.
    score_threshold : float
        Minimum detection confidence for postprocessing.
    """

    def __init__(
        self,
        model_name: str,
        precision: Literal["tf32", "fp16", "bf16"],
        engine_dir: Path,
        force_rebuild: bool = False,
        score_threshold: float = 0.01,
    ) -> None:
        super().__init__(model_name, engine_type="tensorrt", precision=precision)
        self._engine_dir = engine_dir
        self._force_rebuild = force_rebuild
        self._score_threshold = score_threshold
        self._engine_path = engine_dir / f"rtdetr_{precision}.engine"
        self._runtime: object = None
        self._engine: object = None
        self._context: object = None
        self._skipped_reason: str = ""
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._output_shapes: list[tuple[int, ...]] = []

    def load_model(self, weights_path: Path) -> None:
        """Load or build TRT engine from ONNX model.

        Parameters
        ----------
        weights_path : Path
            Path to the .onnx model file.
        """
        self._engine_dir.mkdir(parents=True, exist_ok=True)

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
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        onnx_data = onnx_path.read_bytes()
        if not parser.parse(onnx_data):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            msg = f"ONNX parse failed: {errors}"
            raise RuntimeError(msg)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

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

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            msg = f"TRT engine build failed for precision={self.precision}"
            raise RuntimeError(msg)

        self._engine_path.write_bytes(serialized)
        logger.info(
            "TRT %s engine built and saved: %s", self.precision, self._engine_path
        )

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

        # Cache tensor names and shapes for infer()
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_names.append(name)
                self._output_shapes.append(
                    tuple(self._engine.get_tensor_shape(name))
                )

        logger.info(
            "TRT engine loaded: %s (%.1f MB)",
            self._engine_path.name,
            self.model_size_mb,
        )

    def preprocess(self, sample: COCOSample) -> np.ndarray:
        """Resize image and convert to (1, 3, H, W) float32 numpy array.

        Matches OnnxRuntimeEngine preprocessing: resize to 640x640,
        scale to [0, 1], no ImageNet normalization (RT-DETR convention).
        """
        h, w = _INPUT_SIZE
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
            [logits, pred_boxes] matching OnnxRuntimeEngine output format.
        """
        if self._context is None:
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

        # Execute inference
        self._context.execute_v2([])

        # Copy outputs to CPU
        return [t.cpu().numpy() for t in output_tensors]

    def postprocess(self, raw_outputs: object, sample: COCOSample) -> Detection:
        """Parse TRT RT-DETR outputs to Detection.

        RT-DETR model outputs two tensors:
          - logits: (1, num_queries, num_classes)
          - pred_boxes: (1, num_queries, 4) in [cx, cy, w, h] normalized

        Identical to OnnxRuntimeEngine.postprocess.
        """
        if not isinstance(raw_outputs, list) or len(raw_outputs) < _MIN_TRT_OUTPUTS:
            msg = f"Unexpected TRT output format: {type(raw_outputs)}"
            raise RuntimeError(msg)

        logits: np.ndarray = raw_outputs[0][0]  # (num_queries, num_classes)
        pred_boxes: np.ndarray = raw_outputs[1][0]  # (num_queries, 4) cx cy w h norm

        # Softmax scores + argmax labels
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        scores = probs.max(axis=-1)  # (num_queries,)
        labels = probs.argmax(axis=-1).astype(np.int64)  # (num_queries,)

        # Filter by score threshold
        keep = scores >= self._score_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes_norm = pred_boxes[keep]  # (N, 4) cx cy w h normalized

        # Convert cx cy w h -> x1 y1 x2 y2 in original pixel space
        orig_h, orig_w = sample.original_size
        cx, cy, w, h = (
            boxes_norm[:, 0] * orig_w,
            boxes_norm[:, 1] * orig_h,
            boxes_norm[:, 2] * orig_w,
            boxes_norm[:, 3] * orig_h,
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Labels from model are 0-indexed (80 classes) -> map to COCO 91-class IDs
        coco_labels = np.array(
            [COCO_80_TO_91.get(int(lbl), int(lbl)) for lbl in labels], dtype=np.int64
        )

        return Detection(
            boxes=boxes.reshape(-1, 4),
            scores=scores.astype(np.float32),
            labels=coco_labels,
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
