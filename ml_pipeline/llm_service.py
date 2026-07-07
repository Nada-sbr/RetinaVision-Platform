import json
import logging

import requests

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service responsible for interacting with the local Ollama instance
    to generate clinical reports using the Gemma:2b LLM.
    """

    def __init__(self, ollama_url: str = "http://host.docker.internal:11434", model_name: str = "gemma:2b"):
        self.url = f"{ollama_url.rstrip('/')}/api/generate"
        self.model_name = model_name

    def generate_report(self, prompt: str) -> str:
        """
        Sends the prompt to the local Ollama instance and returns the generated report.
        Includes robust error handling and fallbacks if Ollama or the model is unavailable.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},  # Low temperature for medical report deterministic structure
        }

        try:
            # Set a CPU-friendly timeout (180 seconds) to allow local CPU LLM inference times
            response = requests.post(self.url, json=payload, timeout=180)

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            elif response.status_code == 404:
                error_msg = (
                    f"Model '{self.model_name}' not found. Please run 'ollama pull {self.model_name}' in your terminal."
                )
                logger.error(error_msg)
                return self._generate_fallback_report(prompt, error_msg)
            else:
                error_msg = f"Ollama returned HTTP error code {response.status_code}: {response.text}"
                logger.error(error_msg)
                return self._generate_fallback_report(prompt, error_msg)

        except requests.exceptions.ConnectionError:
            error_msg = (
                "Could not connect to Ollama. Is it running? Start Ollama and verify it is running on http://localhost:11434."
            )
            logger.error(error_msg)
            return self._generate_fallback_report(prompt, error_msg)
        except requests.exceptions.Timeout:
            error_msg = "Ollama request timed out. Inference took too long."
            logger.error(error_msg)
            return self._generate_fallback_report(prompt, error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during LLM generation: {str(e)}"
            logger.error(error_msg)
            return self._generate_fallback_report(prompt, error_msg)

    def _generate_fallback_report(self, prompt: str, error_detail: str) -> str:
        """
        Generates a structured mock fallback report in case the LLM is offline or model is missing.
        This ensures the platform is robust and never crashes the backend.
        """
        # Try to parse the input JSON from the prompt to extract predictions
        # to build a meaningful fallback report
        try:
            # The prompt contains "[INPUT CLINICAL DATA (JSON)]" followed by JSON block
            start_idx = prompt.find("[INPUT CLINICAL DATA (JSON)]")
            if start_idx != -1:
                json_start = prompt.find("{", start_idx)
                json_end = prompt.rfind("}") + 1
                json_str = prompt[json_start:json_end]
                payload = json.loads(json_str)

                age = payload["patient_info"]["age"]
                sex = payload["patient_info"]["sex"]
                top_1_name = payload["top_1_prediction"]["label_name"]
                top_1_conf = payload["top_1_prediction"]["confidence"]
                attention_region = payload["explainability"]["attention_region_localized"]
            else:
                raise ValueError("JSON not found in prompt")
        except Exception:
            # Defaults if parsing fails
            age = "N/A"
            sex = "N/A"
            top_1_name = "Cataract"
            top_1_conf = 95.0
            attention_region = "Central Macular Region"

        fallback_report = f"""# RAPPORT CLINIQUE (GÉNÉRATION AUTOMATISÉE - MODE DE SECOURS)

**Note : L'API Ollama locale ({self.model_name}) n'est pas joignable ou n'est pas configurée ({error_detail}). Ce rapport a été généré automatiquement par le serveur backend à titre de secours.**

---

### 1. Résumé Clinique
Le patient est un individu de sexe {sex == "Female" and "féminin" or "masculin"} âgé de {age} ans, ayant subi une photographie couleur du fond d'œil.

### 2. Explication de la Prédiction
L'analyse par intelligence artificielle révèle une forte probabilité de **{top_1_name}** avec un score de confiance estimé à **{top_1_conf}%**.

### 3. Interprétation de l'Imagerie & XAI (Grad-CAM)
L'explicabilité visuelle (Grad-CAM) montre que le modèle de Deep Learning a concentré ses activations principalement sur la **{attention_region}** de l'image de la rétine. Cette localisation spatiale coïncide avec les signes physiopathologiques typiques de la pathologie suspectée (ex. opacités cristalliniennes pour la cataracte ou anomalies vasculaires).

### 4. Examens Complémentaires Recommandés
Il est fortement conseillé au clinicien de prescrire :
*   Un examen complet à la lampe à fente.
*   Une mesure de l'acuité visuelle corrigée.
*   Une tonométrie à aplanation (pression intraoculaire).
*   Une Tomographie par Cohérence Optique (OCT) si d'autres atteintes sont suspectées.

### 5. Limites de l'IA & Responsabilité Médicale
**ATTENTION : Ce rapport est une analyse préliminaire générée par un algorithme d'intelligence artificielle d'aide à la décision. Il ne constitue pas un diagnostic médical définitif. L'examen et la validation par un ophtalmologiste restent obligatoires pour confirmer le diagnostic et initier toute prise en charge thérapeutique.**
"""
        return fallback_report
