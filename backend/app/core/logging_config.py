import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    """
    Sets up a centralized logging system.
    Configures console output and a rotating file logger for production traceability.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Resolve logs directory path
    base_dir = Path(__file__).resolve().parent.parent.parent
    logs_dir = base_dir / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_file = logs_dir / "app.log"

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs in some envs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Rotating File Handler (Max 10MB per file, keeping last 5 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Log initial startup message
    logging.info("Centralized logging system initialized successfully.")
