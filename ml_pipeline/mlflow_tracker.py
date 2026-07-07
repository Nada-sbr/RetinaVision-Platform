import os
import logging
import mlflow
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wrapper class around MLflow for Ocular AI project experiment tracking
    and model registry integrations.
    """

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "Ocular_Disease_Classification"):
        # Set tracking URI. In Docker, it resolves to http://mlflow:5000
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_name = experiment_name
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(
                f"Could not connect to MLflow server at {self.tracking_uri}: {str(e)}. Running with local tracking."
            )
            # Fallback to local directory tracking
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """Starts a new MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def end_run(self):
        """Ends the active MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Logs parameters dictionary to the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs metrics dictionary to the current run at a given epoch/step."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs a local file or directory as an artifact to the current run."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            logger.error(f"Artifact path does not exist: {local_path}")

    def log_model(self, pytorch_model, artifact_path: str = "model", registered_model_name: Optional[str] = None):
        """Logs a PyTorch model and registers it if name is provided."""
        mlflow.pytorch.log_model(
            pytorch_model=pytorch_model, artifact_path=artifact_path, registered_model_name=registered_model_name
        )
