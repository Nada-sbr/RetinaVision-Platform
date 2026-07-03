import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Automated_Pipeline")

def run_cmd(args):
    """Helper to run system commands and log outputs."""
    logger.info(f"Running command: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(result.stderr)
        return False
    logger.info(result.stdout)
    return True

def run_dvc_pull():
    """Pulls versioned datasets and files from DVC local remote."""
    logger.info("Starting DVC data pull...")
    # Run dvc pull
    return run_cmd(["dvc", "pull"])

def trigger_training():
    """Runs the training pipeline."""
    logger.info("Starting training pipeline...")
    # Add project root to sys.path to ensure correct imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from ml_pipeline.train_effnet import run_pipeline
    try:
        run_pipeline()
        logger.info("Training pipeline executed successfully.")
        return True
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {str(e)}", exc_info=True)
        return False

def main():
    logger.info("========== STARTING AUTOMATED MLOPS PIPELINE ==========")
    
    # 1. Pull data using DVC
    if not run_dvc_pull():
        logger.warning("DVC pull failed or DVC not configured. Proceeding with existing local data.")
    
    # 2. Trigger Model Training
    success = trigger_training()
    
    if success:
        logger.info("========== MLOPS PIPELINE COMPLETED SUCCESSFULLY ==========")
        sys.exit(0)
    else:
        logger.error("========== MLOPS PIPELINE FAILED ==========")
        sys.exit(1)

if __name__ == "__main__":
    main()
