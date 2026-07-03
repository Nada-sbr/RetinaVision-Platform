import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline')))

from model import ODIRResNet50
from gradcam import GradCAM
from report_generator import ClinicalReportGenerator
from transforms import get_val_transforms

def run_report_test():
    print("--- Starting Clinical Report Generator Schema Test ---")
    
    # Paths
    model_path = "checkpoints/best_model.pth"
    sample_img_name = "0_left.jpg"  # Cataract sample in full_df
    img_path = os.path.join("preprocessed_images", sample_img_name)
    
    if not os.path.exists(model_path) or not os.path.exists(img_path):
        print("Required files not found. Creating mock data instead...")
        # Mock data fallback
        patient_age = 69
        patient_sex = "Female"
        mock_predictions = {
            'N': 0.1494,
            'D': 0.1294,
            'G': 0.1684,
            'C': 0.9894,
            'A': 0.0576,
            'H': 0.0769,
            'M': 0.0445,
            'O': 0.4137
        }
        # Create a mock 7x7 heatmap with high values in the center
        mock_heatmap = np.zeros((7, 7))
        mock_heatmap[3, 3] = 1.0
        mock_heatmap[2, 3] = 0.8
        mock_heatmap[4, 3] = 0.8
        
        generator = ClinicalReportGenerator()
        payload = generator.generate_json_payload(patient_age, patient_sex, mock_predictions, mock_heatmap)
        prompt = generator.build_llm_prompt(payload)
    else:
        # Load real data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ODIRResNet50(num_classes=8, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        target_layer = model.backbone.layer4
        gcam = GradCAM(model, target_layer)
        
        original_pil = Image.open(img_path).convert('RGB')
        val_transforms = get_val_transforms(224)
        input_tensor = val_transforms(original_pil).unsqueeze(0).to(device)
        
        # Get real predictions
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
        labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        predictions = {l: float(p) for l, p in zip(labels, probs)}
        
        # Get real heatmap for target class
        class_idx = int(np.argmax(probs))
        heatmap = gcam.generate_heatmap(input_tensor, class_idx)
        
        # Generate payload
        generator = ClinicalReportGenerator()
        payload = generator.generate_json_payload(69, "Female", predictions, heatmap)
        prompt = generator.build_llm_prompt(payload)
        
        gcam.remove_handles()
        
    print("\n================ GENERATED JSON PAYLOAD ================")
    import json
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("========================================================")
    
    print("\n================ GENERATED LLM PROMPT ================")
    print(prompt)
    print("======================================================")
    
    # Save the prompt to a text file for inspection
    os.makedirs("plots", exist_ok=True)
    with open("plots/sample_llm_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\nPrompt file saved to plots/sample_llm_prompt.txt")
    print("Clinical Report Generator Schema Test Completed.")

if __name__ == "__main__":
    run_report_test()
