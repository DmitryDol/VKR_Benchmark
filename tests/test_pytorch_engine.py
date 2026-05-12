from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from benchmark.engines.pytorch_engine import PyTorchEngine


def test_infer_delegation():
    """Verify that PyTorchEngine.infer delegates to ModelAdapter.infer."""
    # Mock dependencies
    mock_adapter = MagicMock()
    # Mock input size for the engine
    mock_adapter.input_size = (640, 640)
    
    mock_model = MagicMock(spec=nn.Module)
    
    # Initialize engine
    # Use 'cpu' device for testing to avoid CUDA requirements
    engine = PyTorchEngine(model_name="test-model", adapter=mock_adapter, device="cpu")
    
    # Manually set the model to avoid load_model logic (TF32 flags)
    engine._model = mock_model
    
    # Input for inference
    mock_inputs = torch.randn(1, 3, 640, 640)
    
    # Run inference
    engine.infer(mock_inputs)
    
    # Assert delegation
    mock_adapter.infer.assert_called_once_with(mock_model, mock_inputs)


def test_infer_raises_if_no_model():
    """Verify that PyTorchEngine.infer raises RuntimeError if model not loaded."""
    mock_adapter = MagicMock()
    engine = PyTorchEngine(model_name="test-model", adapter=mock_adapter, device="cpu")
    
    mock_inputs = torch.randn(1, 3, 640, 640)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.infer(mock_inputs)

if __name__ == "__main__":
    pytest.main([__file__])
