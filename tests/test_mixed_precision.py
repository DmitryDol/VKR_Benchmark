from unittest.mock import MagicMock

import benchmark.engines.mixed_precision as mixed_mod
from benchmark.engines.mixed_precision import (
    apply_strategy_a,
    apply_strategy_b,
    is_constant_or_shape,
)

# Expected FP16-layer counts asserted by the tests below. Named to satisfy PLR2004.
EXPECTED_BOUNDARY_LAYERS = 2
EXPECTED_SOFTMAX_LAYERS_ON_YOLO = 2
EXPECTED_STRATEGY_B_HITS_MIXED = 2


class MockLayerType:
    CONSTANT = 1
    SHAPE = 2
    SOFTMAX = 3
    CONVOLUTION = 4

trt_mock = MagicMock()
trt_mock.LayerType = MockLayerType
trt_mock.float16 = "float16"
mixed_mod.trt = trt_mock

def test_is_constant_or_shape():
    layer = MagicMock()
    layer.type = MockLayerType.CONSTANT
    assert is_constant_or_shape(layer)

def test_apply_strategy_a():
    network = MagicMock()
    network.num_inputs = 1
    network.num_outputs = 1
    network.num_layers = 2

    in_tensor = MagicMock()
    in_tensor.name = "input_0"
    network.get_input.return_value = in_tensor

    out_tensor = MagicMock()
    out_tensor.name = "output_0"
    network.get_output.return_value = out_tensor

    layer0 = MagicMock()
    layer0.type = MockLayerType.CONVOLUTION
    layer0.num_inputs = 1
    layer0.get_input.return_value = in_tensor
    layer0.num_outputs = 1
    layer0.get_output.return_value = MagicMock(name="internal")

    layer1 = MagicMock()
    layer1.type = MockLayerType.CONVOLUTION
    layer1.num_inputs = 1
    layer1.get_input.return_value = MagicMock(name="internal")
    layer1.num_outputs = 1
    layer1.get_output.return_value = out_tensor

    network.get_layer.side_effect = [layer0, layer1]

    count = apply_strategy_a(network)
    assert count == EXPECTED_BOUNDARY_LAYERS
    layer0.set_output_type.assert_called_with(0, "float16")
    layer1.set_output_type.assert_called_with(0, "float16")

def test_apply_strategy_b():
    network = MagicMock()
    network.num_layers = 3

    layer0 = MagicMock()
    layer0.type = MockLayerType.CONVOLUTION
    layer0.name = "conv"

    layer1 = MagicMock()
    layer1.type = MockLayerType.SOFTMAX
    layer1.name = "softmax"

    layer2 = MagicMock()
    layer2.type = MockLayerType.CONVOLUTION
    layer2.name = "LayerNorm"

    network.get_layer.side_effect = [layer0, layer1, layer2]

    count = apply_strategy_b(network)
    assert count == EXPECTED_STRATEGY_B_HITS_MIXED
    assert layer0.precision != "float16"
    layer1.set_output_type.assert_called_with(0, "float16")
    layer2.set_output_type.assert_called_with(0, "float16")


# ---------------------------------------------------------------------------
# Phase 7 Plan 04 — D-13 contract pins for the YOLO/CNN family.
#
# These tests document the Strategy A/B behaviour on a YOLO-shaped (CNN, no
# LayerNorm) network. Per D-13 the heuristic in `mixed_precision.py` is reused
# unchanged for the YOLO family: the SOFTMAX clause catches YOLO11's DFL 16-bin
# softmax / C2PSA attention, and the `"norm"` clause is a documented no-op for
# YOLO11/YOLO26 (CNN family — BatchNorm folded into the conv kernel, not
# LayerNorm). Strategy A still selects the global-IO boundary layers.
# ---------------------------------------------------------------------------


def _make_yolo_layer(layer_type: int, name: str, in_tensor=None, out_tensor=None):
    """Build a mock TRT layer with the minimum surface apply_strategy_* uses."""
    layer = MagicMock()
    layer.type = layer_type
    layer.name = name
    layer.num_inputs = 1 if in_tensor is not None else 0
    layer.num_outputs = 1 if out_tensor is not None else 0
    layer.get_input.return_value = in_tensor
    layer.get_output.return_value = out_tensor
    return layer


def test_strategy_b_selects_softmax_on_yolo_network():
    """YOLO11-shaped graph: SOFTMAX (DFL / C2PSA) layers get FP16, conv stack does not."""
    network = MagicMock()
    network.num_layers = 5

    # Five layers: conv stem, conv backbone, SOFTMAX (DFL), SOFTMAX (C2PSA attention), conv head.
    # No layer name contains "norm" — YOLO11/26 use BatchNorm folded into conv, so the
    # "norm" clause must contribute 0 to the count for this graph (the SOFTMAX clause
    # is the only contributor).
    conv_stem = _make_yolo_layer(MockLayerType.CONVOLUTION, "conv_stem")
    conv_backbone = _make_yolo_layer(MockLayerType.CONVOLUTION, "conv_backbone")
    softmax_dfl = _make_yolo_layer(MockLayerType.SOFTMAX, "dfl_softmax")
    softmax_attn = _make_yolo_layer(MockLayerType.SOFTMAX, "c2psa_attention_softmax")
    conv_head = _make_yolo_layer(MockLayerType.CONVOLUTION, "conv_head")

    network.get_layer.side_effect = [
        conv_stem,
        conv_backbone,
        softmax_dfl,
        softmax_attn,
        conv_head,
    ]

    count = apply_strategy_b(network)

    # Exactly the two SOFTMAX layers are set to FP16.
    assert count == EXPECTED_SOFTMAX_LAYERS_ON_YOLO
    softmax_dfl.set_output_type.assert_called_with(0, "float16")
    softmax_attn.set_output_type.assert_called_with(0, "float16")
    # Convolutions are NOT touched by Strategy B.
    assert conv_stem.set_output_type.call_count == 0
    assert conv_backbone.set_output_type.call_count == 0
    assert conv_head.set_output_type.call_count == 0


def test_strategy_b_norm_clause_is_noop_without_norm_layers():
    """D-13: with no SOFTMAX and no `"norm"`-named layers, Strategy B is a no-op."""
    network = MagicMock()
    network.num_layers = 3

    # Pure-conv CNN graph — no SOFTMAX, no `"norm"` in any layer name. This is the
    # documented worst case for Strategy B on a hypothetical YOLO-shaped model
    # without an attention head: the heuristic returns 0 because both clauses miss.
    conv0 = _make_yolo_layer(MockLayerType.CONVOLUTION, "conv0")
    conv1 = _make_yolo_layer(MockLayerType.CONVOLUTION, "conv1")
    conv2 = _make_yolo_layer(MockLayerType.CONVOLUTION, "head_conv")

    network.get_layer.side_effect = [conv0, conv1, conv2]

    count = apply_strategy_b(network)

    assert count == 0
    assert conv0.set_output_type.call_count == 0
    assert conv1.set_output_type.call_count == 0
    assert conv2.set_output_type.call_count == 0


def test_strategy_a_selects_first_and_last_layers_yolo_shape():
    """Strategy A picks the global-IO boundary layers and skips CONSTANT/SHAPE on a YOLO graph."""
    network = MagicMock()
    network.num_inputs = 1
    network.num_outputs = 1
    network.num_layers = 5

    global_in = MagicMock()
    global_in.name = "images"
    network.get_input.return_value = global_in

    global_out = MagicMock()
    global_out.name = "output0"
    network.get_output.return_value = global_out

    internal_a = MagicMock(name="internal_a")
    internal_a.name = "feat_a"
    internal_b = MagicMock(name="internal_b")
    internal_b.name = "feat_b"
    internal_c = MagicMock(name="internal_c")
    internal_c.name = "feat_c"

    # Layer 0: consumes the global input image -> conv stem (boundary)
    first_conv = _make_yolo_layer(
        MockLayerType.CONVOLUTION, "stem_conv", in_tensor=global_in, out_tensor=internal_a
    )
    # Layer 1: a CONSTANT (anchor / DFL bin weights) — must be skipped
    const_layer = _make_yolo_layer(
        MockLayerType.CONSTANT, "anchor_const", in_tensor=None, out_tensor=internal_b
    )
    # Layer 2: a SHAPE op — must be skipped
    shape_layer = _make_yolo_layer(
        MockLayerType.SHAPE, "shape_op", in_tensor=internal_a, out_tensor=internal_c
    )
    # Layer 3: internal conv — not at the boundary
    mid_conv = _make_yolo_layer(
        MockLayerType.CONVOLUTION, "mid_conv", in_tensor=internal_a, out_tensor=internal_b
    )
    # Layer 4: produces the global output (boundary)
    last_conv = _make_yolo_layer(
        MockLayerType.CONVOLUTION, "head_conv", in_tensor=internal_c, out_tensor=global_out
    )

    network.get_layer.side_effect = [first_conv, const_layer, shape_layer, mid_conv, last_conv]

    count = apply_strategy_a(network)

    # Exactly the two global-IO boundary layers selected; CONSTANT and SHAPE skipped;
    # the internal conv not selected.
    assert count == EXPECTED_BOUNDARY_LAYERS
    first_conv.set_output_type.assert_called_with(0, "float16")
    last_conv.set_output_type.assert_called_with(0, "float16")
    assert const_layer.set_output_type.call_count == 0
    assert shape_layer.set_output_type.call_count == 0
    assert mid_conv.set_output_type.call_count == 0
