import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock tensorrt before imports
trt_mock = MagicMock()
sys.modules["tensorrt"] = trt_mock

from pathlib import Path  # noqa: E402

from benchmark.engines.tensorrt_engine import TensorRTEngine, _BF16UnsupportedError  # noqa: E402


def test_tensorrt_engine_mixed_strategy_paths(tmp_path: Path):
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo11l",
        precision="int8",
        calibrator_method="entropy",
        engine_dir=tmp_path,
        adapter=adapter,
        mixed_strategy="a"
    )
    assert engine._engine_path.name == "yolo11l_mixed_a_entropy.engine"
    assert engine._cache_path.name == "yolo11l_int8_entropy.cache"


def test_tensorrt_engine_model_scoped_non_int8_path(tmp_path: Path):
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo26l",
        precision="fp16",
        engine_dir=tmp_path,
        adapter=adapter,
    )
    assert engine._engine_path.name == "yolo26l_fp16.engine"
    assert engine._cache_path is None


def test_tensorrt_engine_rtdetr_paths_use_model_token(tmp_path: Path):
    """rt-detr model_name uses underscore-sanitized token in filenames."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="rt-detr",
        precision="tf32",
        engine_dir=tmp_path,
        adapter=adapter,
    )
    # Dash sanitized to underscore: rt-detr -> rt_detr
    assert engine._engine_path.name == "rt_detr_tf32.engine"


def test_tensorrt_engine_build_mixed_strategy(tmp_path: Path):
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo11l",
        precision="int8",
        calibrator_method="entropy",
        engine_dir=tmp_path,
        adapter=adapter,
        mixed_strategy="b"
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt, \
         patch("benchmark.engines.tensorrt_engine.TensorRTEngine._apply_int8_config"), \
         patch("benchmark.engines.mixed_precision.apply_strategy_b", return_value=5) as mock_strat:

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = mock_builder.create_network.return_value
        mock_config = mock_builder.create_builder_config.return_value
        mock_builder.build_serialized_network.return_value = b"serialized_data"

        # We need a dummy ONNX file for parse
        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        engine._build_engine(onnx_path)

        mock_config.set_flag.assert_any_call(mock_trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        mock_strat.assert_called_once_with(mock_network)


@pytest.mark.parametrize(("precision", "expected_flag_attr"), [
    ("tf32", "TF32"),
    ("fp16", "FP16"),
])
def test_yolo_trt_build_workspace_and_precision_flags(
    tmp_path: Path,
    precision: str,
    expected_flag_attr: str,
) -> None:
    """TRT build for YOLO models must set 2 GB workspace and correct precision flag."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo11l",
        precision=precision,  # type: ignore[arg-type]
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.build_serialized_network.return_value = b"serialized_data"
        mock_config = mock_builder.create_builder_config.return_value
        # Parser must report success
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        engine._build_engine(onnx_path)

        # strict 2 GB workspace
        mock_config.set_memory_pool_limit.assert_called_once_with(
            mock_trt.MemoryPoolType.WORKSPACE, 2 << 30
        )
        # correct precision flag
        expected_flag = getattr(mock_trt.BuilderFlag, expected_flag_attr)
        mock_config.set_flag.assert_any_call(expected_flag)


def test_yolo_trt_build_bf16_ampere_sets_flag(tmp_path: Path) -> None:
    """BF16 build on Ampere (platform_has_tf32=True) must set BuilderFlag.BF16."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo26l",
        precision="bf16",
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.platform_has_tf32 = True
        mock_builder.build_serialized_network.return_value = b"serialized_data"
        mock_config = mock_builder.create_builder_config.return_value
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        engine._build_engine(onnx_path)

        mock_config.set_flag.assert_any_call(mock_trt.BuilderFlag.BF16)


def test_yolo_trt_build_bf16_non_ampere_raises(tmp_path: Path) -> None:
    """BF16 build on non-Ampere hardware (platform_has_tf32=False) must raise."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="yolo26l",
        precision="bf16",
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.platform_has_tf32 = False
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        with pytest.raises(_BF16UnsupportedError):
            engine._build_engine(onnx_path)


# ---------------------------------------------------------------------------
# RF-DETR-L TensorRT build contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("precision", "expected_flag_attr"), [
    ("tf32", "TF32"),
    ("fp16", "FP16"),
])
def test_rfdetr_l_trt_build_workspace_and_precision_flags(
    tmp_path: Path,
    precision: str,
    expected_flag_attr: str,
) -> None:
    """TRT build for RF-DETR-L must set 2 GB workspace (C-02) and correct precision flag."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="rfdetr-l",
        precision=precision,  # type: ignore[arg-type]
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.build_serialized_network.return_value = b"serialized_data"
        mock_config = mock_builder.create_builder_config.return_value
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        engine._build_engine(onnx_path)

        # C-02: strict 2 GB workspace
        mock_config.set_memory_pool_limit.assert_called_once_with(
            mock_trt.MemoryPoolType.WORKSPACE, 2 << 30
        )
        # C-03 / stage-4: correct precision flag for rfdetr-l
        expected_flag = getattr(mock_trt.BuilderFlag, expected_flag_attr)
        mock_config.set_flag.assert_any_call(expected_flag)


def test_rfdetr_l_trt_build_bf16_ampere_sets_flag(tmp_path: Path) -> None:
    """BF16 build on Ampere RTX 3070 (platform_has_tf32=True) must set BuilderFlag.BF16 (C-04)."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="rfdetr-l",
        precision="bf16",
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.platform_has_tf32 = True
        mock_builder.build_serialized_network.return_value = b"serialized_data"
        mock_config = mock_builder.create_builder_config.return_value
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        engine._build_engine(onnx_path)

        mock_config.set_flag.assert_any_call(mock_trt.BuilderFlag.BF16)


def test_rfdetr_l_trt_build_bf16_non_ampere_raises(tmp_path: Path) -> None:
    """BF16 build on non-Ampere hardware raises _BF16UnsupportedError (C-04)."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="rfdetr-l",
        precision="bf16",
        engine_dir=tmp_path,
        adapter=adapter,
    )

    with patch("benchmark.engines.tensorrt_engine.trt") as mock_trt:
        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_builder.platform_has_tf32 = False
        mock_parser = mock_trt.OnnxParser.return_value
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        onnx_path = tmp_path / "dummy.onnx"
        onnx_path.write_bytes(b"dummy")

        with pytest.raises(_BF16UnsupportedError):
            engine._build_engine(onnx_path)


def test_rfdetr_l_engine_filename_uses_model_token(tmp_path: Path) -> None:
    """rfdetr-l uses underscore-sanitized token in filenames; no collision with rtdetr or yolo."""
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="rfdetr-l",
        precision="tf32",
        engine_dir=tmp_path,
        adapter=adapter,
    )
    # re.sub(r"[^A-Za-z0-9_]", "_", "rfdetr-l") -> "rfdetr_l" (dash -> underscore)
    assert engine._engine_path.name == "rfdetr_l_tf32.engine"
    # Model-scoping: must not collide with RT-DETR or YOLO engine filenames
    assert "rtdetr" not in engine._engine_path.name
    assert "yolo" not in engine._engine_path.name
