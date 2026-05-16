"""Unit tests for scripts/export_rfdetr_onnx.py.

Tests mock rfdetr.RFDETRLarge, simplify_onnx, and validate_onnx so the test
suite runs without a GPU or downloaded weights.

Coverage:
- vendor m.export() called with opset_version=18 and shape=(704, 704)
- project simplify_onnx() called AFTER vendor export (C-10)
- validate_onnx() called AFTER simplify_onnx()
- deprecated `simplify=` kwarg NOT passed to vendor export (regression guard)
- main() returns 0 on success
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import scripts.export_rfdetr_onnx as export_mod

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_OPSET = 18
_EXPECTED_SHAPE = (704, 704)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_and_run(
    tmp_path: Path,
) -> tuple[MagicMock, MagicMock, MagicMock, int]:
    """Patch heavy dependencies, call main(), return mocks + return code.

    Returns
    -------
    (mock_instance, mock_simplify, mock_validate, return_code)
    """
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)

    # Fake ONNX files so main() can call .stat().st_size on them.
    raw_onnx = tmp_path / "inference_model.onnx"
    sim_onnx = tmp_path / "rfdetr_l_sim.onnx"
    raw_onnx.write_bytes(b"x" * (120 * 1024 * 1024 // 1000))  # tiny stand-in
    sim_onnx.write_bytes(b"x" * (118 * 1024 * 1024 // 1000))

    mock_simplify = MagicMock(return_value=sim_onnx)
    mock_validate = MagicMock(return_value=True)

    with (
        patch.object(export_mod, "RFDETRLarge", mock_cls),
        patch.object(export_mod, "simplify_onnx", mock_simplify),
        patch.object(export_mod, "validate_onnx", mock_validate),
        patch.object(sys, "argv", ["export_rfdetr_onnx", "--weights-dir", str(tmp_path)]),
    ):
        rc = export_mod.main()

    return mock_instance, mock_simplify, mock_validate, rc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_main_calls_vendor_export_with_opset_18_and_shape_704(tmp_path: Path) -> None:
    """main() calls m.export(opset_version=18, shape=(704, 704), ...)."""
    mock_instance, _, _, rc = _patch_and_run(tmp_path)

    assert rc == 0
    mock_instance.export.assert_called_once()
    _, kwargs = mock_instance.export.call_args
    assert kwargs.get("opset_version") == _EXPECTED_OPSET, (
        f"Expected opset_version={_EXPECTED_OPSET}, got {kwargs.get('opset_version')}"
    )
    assert kwargs.get("shape") == _EXPECTED_SHAPE, (
        f"Expected shape={_EXPECTED_SHAPE}, got {kwargs.get('shape')}"
    )


def test_main_calls_project_simplify_onnx_after_vendor_export(tmp_path: Path) -> None:
    """main() calls simplify_onnx() AFTER m.export() (C-10 ordering)."""
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)

    raw_onnx = tmp_path / "inference_model.onnx"
    sim_onnx = tmp_path / "rfdetr_l_sim.onnx"
    raw_onnx.write_bytes(b"x" * 1024)
    sim_onnx.write_bytes(b"x" * 512)

    call_order: list[str] = []

    def _export_side_effect(**_kw: object) -> None:
        call_order.append("export")

    def _simplify_side_effect(*_a: object, **_kw: object) -> Path:
        call_order.append("simplify")
        return sim_onnx

    mock_validate = MagicMock(return_value=True)

    with (
        patch.object(export_mod, "RFDETRLarge", mock_cls),
        patch.object(export_mod, "simplify_onnx", side_effect=_simplify_side_effect),
        patch.object(export_mod, "validate_onnx", mock_validate),
        patch.object(sys, "argv", ["export_rfdetr_onnx", "--weights-dir", str(tmp_path)]),
    ):
        mock_instance.export.side_effect = _export_side_effect
        export_mod.main()

    assert "export" in call_order, "m.export() was never called"
    assert "simplify" in call_order, "simplify_onnx() was never called"
    assert call_order.index("export") < call_order.index("simplify"), (
        "vendor m.export() must be called before project simplify_onnx() (C-10)"
    )


def test_main_calls_validate_onnx_after_simplify(tmp_path: Path) -> None:
    """main() calls validate_onnx() AFTER simplify_onnx()."""
    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)

    raw_onnx = tmp_path / "inference_model.onnx"
    sim_onnx = tmp_path / "rfdetr_l_sim.onnx"
    raw_onnx.write_bytes(b"x" * 1024)
    sim_onnx.write_bytes(b"x" * 512)

    call_order: list[str] = []

    def _simplify_side_effect(*_a: object, **_kw: object) -> Path:
        call_order.append("simplify")
        return sim_onnx

    def _validate_side_effect(*_a: object, **_kw: object) -> bool:
        call_order.append("validate")
        return True

    with (
        patch.object(export_mod, "RFDETRLarge", mock_cls),
        patch.object(export_mod, "simplify_onnx", side_effect=_simplify_side_effect),
        patch.object(export_mod, "validate_onnx", side_effect=_validate_side_effect),
        patch.object(sys, "argv", ["export_rfdetr_onnx", "--weights-dir", str(tmp_path)]),
    ):
        export_mod.main()

    assert "simplify" in call_order, "simplify_onnx() was never called"
    assert "validate" in call_order, "validate_onnx() was never called"
    assert call_order.index("simplify") < call_order.index("validate"), (
        "simplify_onnx() must be called before validate_onnx()"
    )


def test_main_does_not_pass_deprecated_simplify_kwarg_to_vendor(tmp_path: Path) -> None:
    """main() does NOT pass simplify= kwarg to vendor m.export().

    The vendor `simplify` kwarg is a deprecated no-op since rfdetr==1.6.
    Passing it logs a vendor warning and is a code smell.
    """
    mock_instance, _, _, _ = _patch_and_run(tmp_path)

    _, kwargs = mock_instance.export.call_args
    assert "simplify" not in kwargs, (
        f"Deprecated 'simplify=' kwarg must NOT be passed to vendor m.export(); "
        f"got kwargs={kwargs!r}"
    )


def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    """main() returns 0 when all mocked calls succeed."""
    _, _, _, rc = _patch_and_run(tmp_path)

    assert rc == 0, f"main() should return 0 on success, got {rc}"
