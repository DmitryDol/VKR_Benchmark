import sys
from unittest.mock import MagicMock, patch

# Mock tensorrt before imports
trt_mock = MagicMock()
sys.modules["tensorrt"] = trt_mock

from pathlib import Path

from benchmark.engines.tensorrt_engine import TensorRTEngine


def test_tensorrt_engine_mixed_strategy_paths(tmp_path: Path):
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="test",
        precision="int8",
        calibrator_method="entropy",
        engine_dir=tmp_path,
        adapter=adapter,
        mixed_strategy="a"
    )
    assert engine._engine_path.name == "rtdetr_mixed_a_entropy.engine"
    assert engine._cache_path.name == "rtdetr_int8_entropy.cache"

def test_tensorrt_engine_build_mixed_strategy(tmp_path: Path):
    adapter = MagicMock()
    engine = TensorRTEngine(
        model_name="test",
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
