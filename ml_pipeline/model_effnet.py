import torch
import torch.nn as nn
import torchvision.models as models

class ODIREfficientNetB3(nn.Module):
    """
    EfficientNet-B3 architecture modified for multi-label classification of ocular diseases.
    Uses pre-trained ImageNet weights and replaces the final classifier block.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super(ODIREfficientNetB3, self).__init__()
        
        if pretrained:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            self.backbone = models.efficientnet_b3(weights=weights)
            print("Loaded EfficientNet-B3 backbone with pre-trained ImageNet weights.")
        else:
            self.backbone = models.efficientnet_b3(weights=None)
            print("Loaded EfficientNet-B3 backbone without pre-trained weights.")
            
        # In torchvision, backbone.classifier is a Sequential block:
        # (0): Dropout(p=0.3, inplace=True)
        # (1): Linear(in_features=1536, out_features=1000, ... )
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom multi-label layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits which are suitable for BCEWithLogitsLoss or FocalLoss.
        """
        return self.backbone(x)
