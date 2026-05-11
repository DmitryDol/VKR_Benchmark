"""
Mixed precision strategies for TensorRT optimization.
"""
import tensorrt as trt
import logging

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
    Strategy B: Apply FP16 to Softmax and LayerNorm nodes.
    """
    count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if is_constant_or_shape(layer):
            continue
            
        if layer.type == trt.LayerType.SOFTMAX or "norm" in layer.name.lower():
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)
            count += 1
            
    return count
