import time
import os
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core import config
from app.core.database import get_db
from app.models import models

router = APIRouter()

# Keep track of start time for uptime calculation
START_TIME = time.time()

# In-memory metrics storage for tracking average latency
# We will update these on inference calls or use default stats
_latency_sum = 0.0
_prediction_count = 0

def log_inference_time(duration_sec: float):
    global _latency_sum, _prediction_count
    _latency_sum += duration_sec
    _prediction_count += 1

@router.get("/health", tags=["monitoring"])
def health_check():
    """
    Standard health check endpoint.
    Used by orchestrators to check if the app process is alive.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/ready", tags=["monitoring"])
def readiness_check(db: Session = Depends(get_db)):
    """
    Checks if the application is fully ready to serve traffic.
    Validates database connectivity and model file existence on disk.
    """
    # Check Database connection
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )

    # Check model checkpoint existence
    model_path = os.path.abspath(config.MODEL_PATH)
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint file missing on disk at {model_path}"
        )

    return {
        "status": "ready",
        "database": "connected",
        "model_loaded": True
    }

@router.get("/metrics", tags=["monitoring"])
def metrics_check(db: Session = Depends(get_db)):
    """
    Exposes application metrics such as database row counts,
    average prediction latency, memory usage, and server uptime.
    """
    global _latency_sum, _prediction_count
    
    # Query database counts
    try:
        total_patients = db.query(models.Patient).count()
        total_predictions_db = db.query(models.Prediction).count()
    except Exception:
        total_patients = 0
        total_predictions_db = 0

    # Calculate latency
    avg_latency = 0.0
    if _prediction_count > 0:
        avg_latency = _latency_sum / _prediction_count

    # System metrics
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "uptime_seconds": time.time() - START_TIME,
        "memory_usage_mb": round(memory_usage_mb, 2),
        "total_patients": total_patients,
        "total_predictions_db": total_predictions_db,
        "in_memory_metrics": {
            "prediction_inferences": _prediction_count,
            "average_latency_sec": round(avg_latency, 4)
        }
    }

@router.get("/version", tags=["monitoring"])
def version_check():
    """
    Exposes application name, version, and current deployment environment.
    """
    return {
        "app_name": config.PROJECT_NAME,
        "version": config.VERSION,
        "environment": config.ENVIRONMENT
    }

@router.get("/model-info", tags=["monitoring"])
def model_info():
    """
    Exposes parameters, target resolution, and network structure details
    about the active PyTorch ML model loaded by the backend.
    """
    return {
        "model_name": "EfficientNet-B3",
        "input_resolution": f"{config.EXPLAIN_RESOLUTION}x{config.EXPLAIN_RESOLUTION}",
        "target_explain_layer": config.EXPLAIN_TARGET_LAYER,
        "classes": ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]
    }

@router.get("/monitoring/drift-report", response_class=HTMLResponse, tags=["monitoring"])
def get_drift_report(db: Session = Depends(get_db)):
    """
    Generates and returns the Evidently AI Data & Prediction Drift report as raw HTML.
    """
    from monitoring.drift_detector import generate_drift_report
    try:
        report_path = generate_drift_report(db)
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drift report generation failed: {str(e)}"
        )
