import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Add ml_pipeline to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline')))

from model import ODIRResNet50
from gradcam import GradCAM
from transforms import get_val_transforms

def run_gradcam_test():
    print("--- Starting Grad-CAM Test ---")
    
    # Paths
    model_path = "checkpoints/best_model.pth"
    sample_img_name = "0_left.jpg"  # Cataract sample in full_df
    img_path = os.path.join("preprocessed_images", sample_img_name)
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Verify files exist
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint {model_path} not found. Train the model first.")
        return
    if not os.path.exists(img_path):
        print(f"Error: Sample image {img_path} not found.")
        return
        
    # 2. Instantiate Model and Load Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ODIRResNet50(num_classes=8, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # 3. Identify Target Layer
    # For ResNet-50, the last convolutional layer is layer4's final residual block's conv3 or the whole layer4.
    # In PyTorch ResNet, model.backbone.layer4 is the final block group. Its output is the final feature maps.
    target_layer = model.backbone.layer4
    print("Target layer selected (model.backbone.layer4).")
    
    # 4. Initialize Grad-CAM
    gcam = GradCAM(model, target_layer)
    
    # 5. Load and Preprocess Image
    original_pil = Image.open(img_path).convert('RGB')
    
    # Convert PIL to Numpy array [H, W, C] for OpenCV overlay
    # OpenCV uses BGR, but we will convert it to RGB.
    original_cv2 = np.array(original_pil)
    
    # Apply standard validation transforms
    val_transforms = get_val_transforms(224)
    input_tensor = val_transforms(original_pil).unsqueeze(0).to(device)
    
    # 6. Run Forward Pass to get predicted probabilities
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    print("\nModel Prediction probabilities:")
    for l, p in zip(labels, probs):
        print(f"  Class {l}: {p:.4f}")
        
    # Pick the class with highest probability as the target
    class_idx = int(np.argmax(probs))
    target_class = labels[class_idx]
    print(f"\nTargeting class index {class_idx} ({target_class}) for Grad-CAM explanation.")
    
    # 7. Generate Heatmap
    heatmap = gcam.generate_heatmap(input_tensor, class_idx)
    
    # 8. Overlay Heatmap on original image
    # Note: original image should be resized to the model input size for better visual alignment,
    # or keep original size. OpenCV handles resizing the heatmap to the original image size.
    colored_heatmap, overlayed_img = gcam.overlay_heatmap(heatmap, original_cv2, alpha=0.4)
    
    # 9. Save visual output
    # Save side-by-side: Original Image, Colored Heatmap, Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_cv2)
    axes[0].set_title("Original Fundus Image")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Raw Heatmap Activations")
    axes[1].axis('off')
    
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f"Grad-CAM Overlay ({target_class})")
    axes[2].axis('off')
    
    plt.suptitle(f"Grad-CAM Explanation for Patient (0_left.jpg) - Class: {target_class}", fontsize=14)
    plt.tight_layout()
    
    demo_save_path = os.path.join(output_dir, "gradcam_demo.png")
    plt.savefig(demo_save_path)
    plt.close()
    
    # Also save separate overlay image
    overlay_save_path = os.path.join(output_dir, "gradcam_overlay_only.png")
    # Convert RGB to BGR for cv2.imwrite
    cv2.imwrite(overlay_save_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
    
    print(f"\nGrad-CAM visual report saved to {demo_save_path}")
    print(f"Grad-CAM overlay image saved to {overlay_save_path}")
    
    # Remove handles
    gcam.remove_handles()
    print("Grad-CAM Test Completed successfully.")

if __name__ == "__main__":
    run_gradcam_test()
