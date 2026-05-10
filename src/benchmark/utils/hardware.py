"""Hardware information collector for benchmark metadata."""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware metadata captured once at CLI startup.

    Fields
    ------
    gpu_name : str
        GPU device name, e.g. "NVIDIA GeForce RTX 3070".
    cuda_version : str
        CUDA runtime version string, e.g. "12.1".
    driver_version : str
        NVIDIA driver version string, e.g. "537.13". Empty string if
        nvidia-smi is unavailable.
    trt_version : str
        TensorRT package version, e.g. "10.16.1.11". Empty string if
        TensorRT is not installed (stages 1-2, per D-02).
    """

    gpu_name: str
    cuda_version: str
    driver_version: str
    trt_version: str  # "" for non-TRT stages — never None (D-02)

    @classmethod
    def collect(cls) -> HardwareInfo:
        """Query GPU, CUDA, driver, and TRT versions from the system.

        Calls nvidia-smi via subprocess with a fixed argument list
        (no shell=True, no user input — T-02-01 mitigation).
        TRT version falls back to "" if package not installed (D-02).

        Returns
        -------
        HardwareInfo
            Populated hardware metadata instance.
        """
        # GPU name via torch
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )

        # CUDA version via torch.version.cuda
        cuda_version: str = torch.version.cuda or ""  # type: ignore[attr-defined]

        # Driver version via nvidia-smi (fixed arg list — no shell injection)
        driver_version = cls._query_driver_version()

        # TRT version via importlib.metadata — "" if not installed (D-02)
        trt_version = cls._query_trt_version()

        info = cls(
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            driver_version=driver_version,
            trt_version=trt_version,
        )
        logger.info(
            "Hardware: %s | CUDA %s | Driver %s | TRT %s",
            info.gpu_name,
            info.cuda_version,
            info.driver_version or "unknown",
            info.trt_version or "not installed",
        )
        return info

    @staticmethod
    def _query_driver_version() -> str:
        """Query NVIDIA driver version via nvidia-smi.

        Uses a fixed argument list (T-02-01 mitigation: no shell=True,
        no user-controlled args).

        Returns empty string if nvidia-smi is unavailable or errors.
        """
        try:
            cmd = [  # fixed list, no user input — T-02-01 mitigation
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip().splitlines()[0].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.warning("nvidia-smi unavailable — driver_version will be empty")
        return ""

    @staticmethod
    def _query_trt_version() -> str:
        """Return TensorRT package version or empty string if not installed."""
        try:
            return importlib.metadata.version("tensorrt")
        except importlib.metadata.PackageNotFoundError:
            return ""
