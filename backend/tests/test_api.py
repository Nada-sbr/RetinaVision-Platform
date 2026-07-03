import pytest
import io

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["health_check"] == "passed"

def test_register_user(client):
    response = client.post(
        "/api/register",
        json={
            "email": "doctor@test.com",
            "password": "securepassword123",
            "full_name": "Dr. Test User",
            "role": "ophthalmologist"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "doctor@test.com"
    assert data["full_name"] == "Dr. Test User"
    assert "id" in data
    assert "hashed_password" not in data

def test_login_user(client):
    # 1. Register first
    client.post(
        "/api/register",
        json={
            "email": "doctor2@test.com",
            "password": "securepassword123",
            "full_name": "Dr. Test User 2"
        }
    )
    
    # 2. Login
    response = client.post(
        "/api/login",
        data={
            "username": "doctor2@test.com",
            "password": "securepassword123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_create_patient_unauthorized(client):
    response = client.post(
        "/api/patients",
        json={
            "first_name": "Jean",
            "last_name": "Dupont",
            "birth_date": "1980-01-01",
            "gender": "Male"
        }
    )
    assert response.status_code == 401

def test_create_patient_authorized(client):
    # 1. Register and Login
    client.post(
        "/api/register",
        json={
            "email": "doctor3@test.com",
            "password": "securepassword123",
            "full_name": "Dr. Test User 3"
        }
    )
    login_res = client.post(
        "/api/login",
        data={"username": "doctor3@test.com", "password": "securepassword123"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Create patient
    response = client.post(
        "/api/patients",
        json={
            "first_name": "Marie",
            "last_name": "Curie",
            "birth_date": "1867-11-07",
            "gender": "Female"
        },
        headers=headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["first_name"] == "Marie"
    assert data["last_name"] == "Curie"
    assert data["gender"] == "Female"
    assert "id" in data

def test_health_endpoint(client):
    """Test health check route."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

def test_ready_endpoint(client, db):
    """Test ready check route. Mocks model checkpoint existence to verify success/failure."""
    import os
    from app.core import config
    
    original_model_path = config.MODEL_PATH
    
    # 1. Test when model file does not exist
    config.MODEL_PATH = "nonexistent_model.pth"
    response = client.get("/ready")
    assert response.status_code == 503
    assert "missing" in response.json()["detail"]
    
    # 2. Test when model file exists (temporary dummy file creation)
    config.MODEL_PATH = "temp_dummy_test_model.pth"
    with open(config.MODEL_PATH, "w") as f:
        f.write("dummy model content")
        
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["model_loaded"] is True
    finally:
        if os.path.exists(config.MODEL_PATH):
            os.remove(config.MODEL_PATH)
        config.MODEL_PATH = original_model_path

def test_metrics_endpoint(client):
    """Test metrics monitoring route."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "memory_usage_mb" in data
    assert "total_patients" in data
    assert "total_predictions_db" in data
    assert "in_memory_metrics" in data

def test_version_endpoint(client):
    """Test API version details route."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data

def test_model_info_endpoint(client):
    """Test model metadata query route."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "EfficientNet-B3"
    assert "input_resolution" in data
    assert "classes" in data

def test_drift_report_endpoint(client, monkeypatch):
    """Test Evidently AI drift report generation endpoint returns HTML."""
    # Mock out the heavy, segfault-prone Evidently calculations during testing
    def mock_generate_drift_report(db):
        temp_html = "temp_drift_test.html"
        with open(temp_html, "w", encoding="utf-8") as f:
            f.write("<html>Evidently Drift Report Mock</html>")
        return temp_html

    import monitoring.drift_detector
    monkeypatch.setattr(monitoring.drift_detector, "generate_drift_report", mock_generate_drift_report)

    try:
        response = client.get("/monitoring/drift-report")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Evidently" in response.text
    finally:
        import os
        if os.path.exists("temp_drift_test.html"):
            os.remove("temp_drift_test.html")
