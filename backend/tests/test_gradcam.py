import pytest
import sys
import os
import torch
import numpy as np

# Add pipeline path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_pipeline.model_effnet import ODIREfficientNetB3
from ml_pipeline.gradcam import GradCAM

def test_gradcam_generation():
    """Verify Grad-CAM registers hooks and generates heatmaps properly."""
    model = ODIREfficientNetB3(num_classes=8, pretrained=False)
    target_layer = model.backbone.features[8]
    
    gcam = GradCAM(model, target_layer)
    dummy_input = torch.randn(1, 3, 300, 300)
    
    # Generate heatmap for class 0
    heatmap = gcam.generate_heatmap(dummy_input, class_idx=0)
    
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (10, 10)  # Raw activation map size for B3 final layer
    
    # Verify overlay resizing to (300, 300, 3)
    dummy_original = np.zeros((300, 300, 3), dtype=np.uint8)
    heatmap_colored, overlayed_img = gcam.overlay_heatmap(heatmap, dummy_original)
    gcam.remove_handles()
    
    assert overlayed_img.shape == (300, 300, 3)
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
