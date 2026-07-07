import json
from typing import Any, Dict

import numpy as np


class ClinicalReportGenerator:
    """
    Generates structured JSON payloads from prediction metadata and Grad-CAM heatmaps,
    and builds optimized prompts for the clinical report LLM.
    """

    def __init__(self, labels_mapping: Dict[str, str] = None):
        # Human-readable mapping of ODIR labels
        self.labels_mapping = labels_mapping or {
            "N": "Normal Fundus",
            "D": "Diabetic Retinopathy",
            "G": "Glaucoma",
            "C": "Cataract",
            "A": "Age-related Macular Degeneration (AMD)",
            "H": "Hypertensive Retinopathy",
            "M": "Pathological Myopia",
            "O": "Other Disease / Abnormality",
        }

    def localize_activation_region(self, heatmap: np.ndarray, threshold: float = 0.5) -> str:
        """
        Analyzes the Grad-CAM heatmap to localize the region of highest model attention
        mapping it to a 3x3 spatial grid.
        """
        # Threshold the heatmap to keep only high-intensity regions
        binary_map = (heatmap >= threshold).astype(np.uint8)

        # Compute coordinates of the active pixels
        y_indices, x_indices = np.where(binary_map > 0)

        if len(x_indices) == 0 or len(y_indices) == 0:
            return "Diffuse / Unlocalized"

        # Calculate the centroid (center of mass) of the active region
        mean_y = np.mean(y_indices) / heatmap.shape[0]  # normalized [0, 1]
        mean_x = np.mean(x_indices) / heatmap.shape[1]  # normalized [0, 1]

        # Map to 3x3 grid
        # Y-axis divisions
        if mean_y < 0.33:
            y_pos = "Superior"
        elif mean_y > 0.66:
            y_pos = "Inferior"
        else:
            y_pos = "Central"

        # X-axis divisions
        if mean_x < 0.33:
            x_pos = "Left / Temporal"
        elif mean_x > 0.66:
            x_pos = "Right / Nasal"
        else:
            x_pos = "Center"

        # Combine descriptions
        if y_pos == "Central" and x_pos == "Center":
            return "Central Macular Region"
        elif y_pos == "Central":
            return f"Central-Lateral ({x_pos}) Region"
        elif x_pos == "Center":
            return f"{y_pos}-Vertical Region"
        else:
            return f"{y_pos}-{x_pos} Quadrent"

    def generate_json_payload(
        self, patient_age: int, patient_sex: str, predictions: Dict[str, float], heatmap: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compiles patient details, prediction scores, and spatial heatmap localization
        into a structured dictionary (JSON payload).

        Args:
            patient_age (int): Patient age.
            patient_sex (str): Patient sex.
            predictions (dict): Dictionary mapping ODIR class label string to float probability.
            heatmap (np.ndarray): 2D Grad-CAM heatmap array.
        """
        # Sort predictions by confidence
        sorted_preds = sorted(predictions.items(), key=lambda item: item[1], reverse=True)

        top_1_label, top_1_prob = sorted_preds[0]

        # Get top 3 predictions
        top_3 = []
        for label, prob in sorted_preds[:3]:
            top_3.append(
                {
                    "label_code": label,
                    "label_name": self.labels_mapping.get(label, "Unknown"),
                    "confidence": round(float(prob) * 100, 2),  # Percentage representation
                }
            )

        # Localize Grad-CAM region
        attention_region = self.localize_activation_region(heatmap)

        payload = {
            "patient_info": {"age": patient_age, "sex": patient_sex},
            "top_1_prediction": {
                "label_code": top_1_label,
                "label_name": self.labels_mapping.get(top_1_label, "Unknown"),
                "confidence": round(float(top_1_prob) * 100, 2),
            },
            "top_3_predictions": top_3,
            "explainability": {
                "method": "Grad-CAM",
                "target_layer": "backbone.layer4",
                "attention_region_localized": attention_region,
            },
        }

        return payload

    def build_llm_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Constructs the system prompt and instructions for the Mistral LLM
        based on the structured JSON payload.
        """
        payload["patient_info"]
        top_1 = payload["top_1_prediction"]
        payload["top_3_predictions"]
        xai = payload["explainability"]

        # Build prompt using markdown structure
        prompt = f"""[SYSTEM INSTRUCTIONS]
You are a highly experienced Senior Ophthalmologist and Clinical AI Decision Support Assistant.
Your task is to generate a structured, professional clinical report based on the provided patient metadata and deep learning prediction results.

Rules:
1. Always write the report in French.
2. Your report must act as a clinical decision support tool. It MUST NEVER state a definitive, final diagnosis.
3. You must include a mandatory warning stating that this is an AI tool and the final medical decision belongs to the attending physician.
4. Maintain a highly professional, academic, and clinical tone.

[INPUT CLINICAL DATA (JSON)]
{json.dumps(payload, indent=2, ensure_ascii=False)}

[REPORT TEMPLATE STRUCTURE]
Please construct the clinical report following this exact section structure:

1. **Résumé Clinique (Clinical Summary)**:
   - Present the patient (age, sex) and the context of the fundus exam.

2. **Explication de la Prédiction (Prediction Explanation)**:
   - Detail the primary prediction (Top-1) and confidence score. Mention the other likely differentials in the Top-3.

3. **Interprétation de l'Imagerie (Image Interpretation & XAI)**:
   - Interpret the Grad-CAM explanation. Discuss the significance of the model's focus on the '{xai["attention_region_localized"]}' region of the fundus. Connect this spatial focus to the pathophysiology of '{top_1["label_name"]}'.

4. **Examens Complémentaires Recommandés (Recommended Examinations)**:
   - Suggest relevant clinical follow-up tests (e.g., OCT, Optical Coherence Tomography, angiography, visual field, intraocular pressure measurement) depending on the predicted pathology.

5. **Limites de l'IA & Responsabilité Médicale (AI Limitations & Legal Warning)**:
   - Provide a clear, bold disclaimer that the AI prediction is not a final diagnosis. The physician's clinical verification is mandatory.

Generate the clinical report now:
"""
        return prompt
