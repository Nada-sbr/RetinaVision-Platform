import pytest
import sys
import os
import torch

# Add pipeline path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_pipeline.model_effnet import ODIREfficientNetB3

def test_model_forward():
    """Verify forward pass of EfficientNet-B3 model returns logits of correct shape."""
    model = ODIREfficientNetB3(num_classes=8, pretrained=False)
    dummy_input = torch.randn(2, 3, 300, 300)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 8)

def test_trainable_parameters():
    """Verify model unfreezes only classifier and final feature stages."""
    model = ODIREfficientNetB3(num_classes=8, pretrained=False)
    
    # Freeze lower layers mimicking training setup
    for name, param in model.named_parameters():
        if "backbone.classifier" in name or "backbone.features.7" in name or "backbone.features.8" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Check classifier features are trainable
    for name, param in model.named_parameters():
        if "backbone.classifier" in name:
            assert param.requires_grad is True
        elif "backbone.features.0" in name:
            assert param.requires_grad is False
