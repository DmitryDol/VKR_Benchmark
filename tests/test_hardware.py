"""Tests for HardwareInfo.collect() using mocks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmark.utils.hardware import HardwareInfo


def test_collect_returns_hardware_info() -> None:
    """HardwareInfo.collect() must return a HardwareInfo instance."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
        patch("torch.version.cuda", "12.1"),
        patch.object(HardwareInfo, "_query_driver_version", return_value="537.13"),
        patch.object(HardwareInfo, "_query_trt_version", return_value=""),
    ):
        hw = HardwareInfo.collect()

    assert isinstance(hw, HardwareInfo)
    assert hw.gpu_name == "Test GPU"
    assert hw.cuda_version == "12.1"
    assert hw.driver_version == "537.13"
    assert hw.trt_version == ""


def test_trt_version_empty_when_not_installed() -> None:
    """hw_trt_version must be '' (not None) when TRT is not installed (D-02)."""
    import importlib.metadata

    with patch(
        "importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("tensorrt"),
    ):
        version = HardwareInfo._query_trt_version()
    assert version == ""
    assert version is not None


def test_driver_version_empty_on_subprocess_failure() -> None:
    """driver_version must be '' when nvidia-smi is unavailable."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        version = HardwareInfo._query_driver_version()
    assert version == ""


def test_driver_version_uses_fixed_arg_list() -> None:
    """nvidia-smi call must use a fixed arg list, not shell=True (T-02-01)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "537.13\n"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        HardwareInfo._query_driver_version()

    call_kwargs = mock_run.call_args
    # shell must not be True
    shell_arg = call_kwargs.kwargs.get("shell", False)
    assert shell_arg is not True, "subprocess.run must not use shell=True"
    # first positional arg must be a list
    first_arg = call_kwargs.args[0]
    assert isinstance(first_arg, list), "subprocess.run must use list args, not string"
    assert "nvidia-smi" in first_arg[0]


def test_collect_cpu_fallback_when_cuda_unavailable() -> None:
    """When CUDA is unavailable, gpu_name must be 'CPU'."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.version.cuda", None),
        patch.object(HardwareInfo, "_query_driver_version", return_value=""),
        patch.object(HardwareInfo, "_query_trt_version", return_value=""),
    ):
        hw = HardwareInfo.collect()
    assert hw.gpu_name == "CPU"
