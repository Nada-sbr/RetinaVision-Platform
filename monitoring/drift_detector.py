import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

try:
    from evidently.report import Report
except ImportError:
    from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    from evidently.legacy.metric_preset import DataDriftPreset

from app.models import models


def fetch_production_predictions(db: Session) -> pd.DataFrame:
    """Fetches the latest prediction probabilities from the database."""
    predictions = db.query(models.Prediction).all()
    if not predictions:
        # Return empty df with expected structure if no predictions exist
        return pd.DataFrame(columns=["n_prob", "d_prob", "g_prob", "c_prob", "a_prob", "h_prob", "m_prob", "o_prob"])

    data = []
    for pred in predictions:
        data.append([pred.n_prob, pred.d_prob, pred.g_prob, pred.c_prob, pred.a_prob, pred.h_prob, pred.m_prob, pred.o_prob])

    return pd.DataFrame(data, columns=["n_prob", "d_prob", "g_prob", "c_prob", "a_prob", "h_prob", "m_prob", "o_prob"])


def get_reference_predictions() -> pd.DataFrame:
    """
    Returns baseline reference predictions (e.g., from training/validation logs).
    In this case, we mock a stable distribution similar to baseline evaluation metrics.
    """
    np.random.seed(42)
    n_samples = 100
    # Simulate realistic prediction values from training
    data = {
        "n_prob": np.random.beta(5, 2, n_samples),
        "d_prob": np.random.beta(2, 5, n_samples),
        "g_prob": np.random.beta(2, 5, n_samples),
        "c_prob": np.random.beta(2, 5, n_samples),
        "a_prob": np.random.beta(1, 5, n_samples),
        "h_prob": np.random.beta(1, 5, n_samples),
        "m_prob": np.random.beta(1, 5, n_samples),
        "o_prob": np.random.beta(2, 5, n_samples),
    }
    return pd.DataFrame(data)


def generate_drift_report(db: Session) -> str:
    """
    Compares production predictions with reference predictions using Evidently AI
    and writes the resulting HTML report to disk.
    """
    reference = get_reference_predictions()
    current = fetch_production_predictions(db)

    # If current data is too small, inject a small amount of simulated data
    # to allow Evidently AI calculations to run without failing
    if len(current) < 5:
        # Create a tiny mock current dataset mixed with real data
        mock_current = get_reference_predictions().iloc[:5].copy()
        if len(current) > 0:
            mock_current.iloc[: len(current)] = current
        current = mock_current

    # Setup report directory
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "monitoring" / "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = reports_dir / "drift_report.html"

    # Define Evidently report
    data_drift_report = Report(metrics=[DataDriftPreset()])

    data_drift_report.run(reference_data=reference, current_data=current)
    data_drift_report.save_html(str(report_file))

    return str(report_file)
