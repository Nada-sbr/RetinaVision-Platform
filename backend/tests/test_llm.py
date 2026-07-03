import pytest
import sys
import os
import requests
import requests_mock

# Add pipeline path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_pipeline.llm_service import LLMService

def test_llm_service_success():
    """Verify LLMService processes successful API responses."""
    service = LLMService(ollama_url="http://fake-ollama:11434", model_name="gemma:2b")
    
    with requests_mock.Mocker() as m:
        m.post(
            "http://fake-ollama:11434/api/generate",
            json={"response": "This is a clinical report for glaucoma."}
        )
        report = service.generate_report("Glaucoma analysis")
        
    assert report == "This is a clinical report for glaucoma."

def test_llm_service_connection_error():
    """Verify LLMService fails gracefully into fallback report on connection loss."""
    service = LLMService(ollama_url="http://fake-ollama:11434", model_name="gemma:2b")
    
    with requests_mock.Mocker() as m:
        m.post(
            "http://fake-ollama:11434/api/generate",
            exc=requests.exceptions.ConnectionError
        )
        report = service.generate_report("Glaucoma analysis")
        
    # Should fall back to warning report
    assert "RAPPORT CLINIQUE (GÉNÉRATION AUTOMATISÉE - MODE DE SECOURS)" in report
    assert "Could not connect to Ollama" in report

def test_llm_service_timeout():
    """Verify LLMService fails gracefully into fallback report on timeout."""
    service = LLMService(ollama_url="http://fake-ollama:11434", model_name="gemma:2b")
    
    with requests_mock.Mocker() as m:
        m.post(
            "http://fake-ollama:11434/api/generate",
            exc=requests.exceptions.Timeout
        )
        report = service.generate_report("Glaucoma analysis")
        
    assert "RAPPORT CLINIQUE (GÉNÉRATION AUTOMATISÉE - MODE DE SECOURS)" in report
    assert "timed out" in report
