from unittest.mock import MagicMock, patch
from pathlib import Path
from benchmark.cli import _run_stage

def test_cli_mixed_precision_stage(tmp_path: Path):
    result_logger = MagicMock()
    result_logger.output_dir = tmp_path
    result_logger.run_id = "test_run"
    
    # Create mock calibrator file
    cal_file_dir = tmp_path / "rt-detr" / "test_run"
    cal_file_dir.mkdir(parents=True)
    cal_file = cal_file_dir / "int8_best_calibrator.json"
    cal_file.write_text('{"best_calibrator": "percentile"}', encoding="utf-8")
    
    # Mock ONNX file
    import benchmark.cli as cli_mod
    cli_mod.MODEL_REGISTRY = {
        "rt-detr": {
            "weights": "weights",
            "onnx": str(tmp_path / "dummy.onnx")
        }
    }
    (tmp_path / "dummy.onnx").write_text("dummy")
    
    with patch("benchmark.cli.TensorRTEngine") as mock_engine_cls, \
         patch("benchmark.cli.COCODataLoader") as mock_loader:
        
        mock_engine = mock_engine_cls.return_value
        mock_engine.run_full_benchmark.return_value.map_50_95 = 0.5
        
        _run_stage(
            model_name="rt-detr",
            stage="6_trt_mixed_a",
            limit=10,
            result_logger=result_logger,
            baseline_map=0.0,
            macs=0.0,
            flops=0.0,
            engine_dir=tmp_path
        )
        
        # Check that it called TensorRTEngine with correct args
        mock_engine_cls.assert_called_with(
            model_name="rt-detr",
            precision="int8",
            calibrator_method="percentile",
            engine_dir=tmp_path,
            force_rebuild=False,
            mixed_strategy="a"
        )
