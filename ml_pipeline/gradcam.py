from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation in PyTorch.
    Extracts heatmaps from the final convolutional layer of a CNN model to explain predictions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        # Placeholders for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Capture the forward activations
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Capture the backward gradients of the output w.r.t activations
            self.gradients = grad_output[0].detach()

        # Hook into the target layer
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates the raw 2D Grad-CAM heatmap for a given input tensor and class index.
        """
        # Ensure tensor is on the correct device and has a batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(next(self.model.parameters()).device)

        # 1. Forward pass
        logits = self.model(input_tensor)
        score = logits[0, class_idx]

        # 2. Backward pass
        self.model.zero_grad()
        score.backward()

        # 3. Calculate weights (alpha_c_k) via Global Average Pooling
        # Gradients shape: [batch, channels, height, width]
        gradients = self.gradients[0]  # Shape: [channels, H, W]
        activations = self.activations[0]  # Shape: [channels, H, W]

        # Global Average Pooling of gradients over spatial dimensions H and W
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: [channels]

        # 4. Weighted combination of activation maps
        # Initialize heatmap with zeros, same spatial shape as activation maps
        h, w = activations.shape[1], activations.shape[2]
        heatmap = torch.zeros((h, w), dtype=torch.float32, device=activations.device)

        for k in range(len(weights)):
            heatmap += weights[k] * activations[k]

        # 5. Apply ReLU activation to keep only features that positively influence target class score
        heatmap = torch.clamp(heatmap, min=0)

        # 6. Normalize heatmap between 0 and 1
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        # Convert to numpy array
        return heatmap.cpu().numpy()

    def overlay_heatmap(
        self, heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize heatmap, apply colormap, and overlay on the original image.

        Args:
            heatmap (np.ndarray): 2D heatmap normalized [0, 1].
            original_img (np.ndarray): Original RGB image (numpy array shape HxWx3, values [0, 255]).
            alpha (float): Transparency factor of the heatmap overlay.
            colormap (int): OpenCV colormap.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (colored_heatmap, overlayed_img)
        """
        # Get dimensions
        h, w = original_img.shape[0], original_img.shape[1]

        # Resize heatmap to match the original image size using bilinear interpolation
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert heatmap to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap (translates single channel heatmap into colored heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Convert OpenCV colored heatmap (BGR) to RGB to match original image
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on original image: output = alpha * heatmap + (1 - alpha) * original
        overlayed_img = cv2.addWeighted(heatmap_colored_rgb, alpha, original_img, 1.0 - alpha, 0)

        return heatmap_colored_rgb, overlayed_img

    def remove_handles(self):
        """
        Removes the hooks from the model to prevent memory leaks.
        """
        self.forward_handle.remove()
        self.backward_handle.remove()
