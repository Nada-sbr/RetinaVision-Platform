import pytest
import sys
import os
from PIL import Image
import torch

# Add pipeline path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_pipeline.transforms import get_train_transforms, get_val_transforms

def test_train_transforms():
    """Verify train transforms outputs correct dimensions."""
    img = Image.new("RGB", (500, 500), color="white")
    transform = get_train_transforms(300)
    transformed_img = transform(img)
    
    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (3, 300, 300)

def test_val_transforms():
    """Verify validation transforms outputs correct dimensions."""
    img = Image.new("RGB", (500, 500), color="white")
    transform = get_val_transforms(300)
    transformed_img = transform(img)
    
    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (3, 300, 300)
