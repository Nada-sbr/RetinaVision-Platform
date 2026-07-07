import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_pipeline")))

from llm_service import LLMService


def run_llm_test():
    print("--- Starting LLM Service Test (Ollama - Gemma:2b) ---")

    prompt_path = "plots/sample_llm_prompt.txt"
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file {prompt_path} not found. Run test_report.py first.")
        return

    # Read the prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    # Instantiate LLM Service
    llm = LLMService(ollama_url="http://localhost:11434", model_name="gemma:2b")

    print("\nSending prompt to Ollama (Gemma:2b)... Please wait (this can take 5-20 seconds)...")
    report = llm.generate_report(prompt)

    print("\n================ GENERATED CLINICAL REPORT ================")
    print(report)
    print("===========================================================")

    # Save the report
    os.makedirs("plots", exist_ok=True)
    report_save_path = "plots/clinical_report_output.md"
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nClinical report saved to {report_save_path}")
    print("LLM Service Test Completed.")


if __name__ == "__main__":
    run_llm_test()
