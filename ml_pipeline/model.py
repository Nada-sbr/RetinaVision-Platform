import torch
import torch.nn as nn
import torchvision.models as models

class ODIRResNet50(nn.Module):
    """
    ResNet-50 architecture modified for multi-label classification of ocular diseases.
    Uses pre-trained ImageNet weights and replaces the final fully connected layer.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super(ODIRResNet50, self).__init__()
        
        # Load ResNet-50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
            print("Loaded ResNet-50 backbone with pre-trained ImageNet weights.")
        else:
            self.backbone = models.resnet50(weights=None)
            print("Loaded ResNet-50 backbone without pre-trained weights.")
            
        # Extract features size of the original fc layer
        in_features = self.backbone.fc.in_features
        
        # Replace the original fc layer with a custom classifier block
        # We include a Dropout layer (p=0.5) to prevent overfitting during transfer learning
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns logits (before Sigmoid activation) which is standard for BCEWithLogitsLoss.
        """
        return self.backbone(x)
