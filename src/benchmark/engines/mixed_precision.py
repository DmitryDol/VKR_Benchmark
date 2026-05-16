"""
Mixed precision strategies for TensorRT optimization.
"""
import logging

import tensorrt as trt

logger = logging.getLogger(__name__)

def is_constant_or_shape(layer: trt.ILayer) -> bool:
    """Check if a layer is CONSTANT or SHAPE."""
    return layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE)

def apply_strategy_a(network: trt.INetworkDefinition) -> int:
    """
    Strategy A: Apply FP16 to the first and last layers connected to global IO.
    """
    global_inputs = {network.get_input(i).name for i in range(network.num_inputs)}
    global_outputs = {network.get_output(i).name for i in range(network.num_outputs)}

    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if is_constant_or_shape(layer):
            continue

        is_boundary = False
        # Check inputs
        for j in range(layer.num_inputs):
            inp = layer.get_input(j)
            if inp and inp.name in global_inputs:
                is_boundary = True
                break

        # Check outputs
        if not is_boundary:
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out and out.name in global_outputs:
                    is_boundary = True
                    break

        if is_boundary:
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
            count += 1

    return count

def apply_strategy_b(network: trt.INetworkDefinition) -> int:
    """
    Strategy B: Apply FP16 to Softmax and Normalization (LayerNorm) nodes.

    Marks layers as FP16 if any of:
      - layer.type == LayerType.SOFTMAX (attention Softmax)
      - layer.type == LayerType.NORMALIZATION (INormalizationLayer / LayerNorm /
        GroupNorm, TRT 8.6+)
      - 'norm' in layer.name.lower() (substring fallback for graphs where the
        opset < 17 decomposed LayerNorm or used a different naming scheme)

    Phase 8 RF-DETR (D-RF-03 = B2): the explicit LayerType.NORMALIZATION clause
    hardens the contract against future graphs whose node names don't contain
    'norm'. Carries forward unchanged to Phase 10 (D-FINE, DEIMv2).
    """
    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if is_constant_or_shape(layer):
            continue

        # D-RF-03 B2: NORMALIZATION clause added to the type set so
        # INormalizationLayer (TRT 8.6+ native LayerNorm) is caught
        # name-agnostically alongside SOFTMAX. The substring fallback
        # still covers opset<17 decomposed-LayerNorm graphs.
        if layer.type in {trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION} \
                or "norm" in layer.name.lower():
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
            count += 1

    return count
