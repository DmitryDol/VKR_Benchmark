from unittest.mock import MagicMock

import benchmark.engines.mixed_precision as mixed_mod
from benchmark.engines.mixed_precision import (
    apply_strategy_a,
    apply_strategy_b,
    is_constant_or_shape,
)


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
    assert count == 2
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
    assert count == 2
    assert layer0.precision != "float16"
    layer1.set_output_type.assert_called_with(0, "float16")
    layer2.set_output_type.assert_called_with(0, "float16")
