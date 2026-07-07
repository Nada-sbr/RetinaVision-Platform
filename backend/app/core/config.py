import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env if present
# Search from current file path up to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Load YAML configuration
CONFIG_PATH = BASE_DIR / "configs" / "app.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        _yaml_config = yaml.safe_load(f) or {}
else:
    _yaml_config = {}


# Config helper functions
def get_yaml_val(keys: List[str], default=None):
    curr = _yaml_config
    for key in keys:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return default
    return curr


# Expose Config Parameters
ENVIRONMENT = os.getenv("ENVIRONMENT", get_yaml_val(["app", "environment"], "production"))
PROJECT_NAME = get_yaml_val(["app", "name"], "Ocular AI Platform")
VERSION = get_yaml_val(["app", "version"], "1.0.0")

# API settings
API_HOST = os.getenv("API_HOST", get_yaml_val(["api", "host"], "0.0.0.0"))
API_PORT = int(os.getenv("API_PORT", get_yaml_val(["api", "port"], 8000)))

# CORS origins
raw_origins = os.getenv("CORS_ORIGINS")
if raw_origins:
    CORS_ORIGINS = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]
else:
    CORS_ORIGINS = get_yaml_val(["api", "cors_origins"], ["http://localhost", "http://127.0.0.1"])

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/ocular_db")
DB_POOL_SIZE = int(get_yaml_val(["database", "pool_size"], 20))
DB_MAX_OVERFLOW = int(get_yaml_val(["database", "max_overflow"], 10))
DB_POOL_PRE_PING = bool(get_yaml_val(["database", "pool_pre_ping"], True))

# ML model settings
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

# Explainability settings
EXPLAIN_TARGET_LAYER = get_yaml_val(["explainability", "target_layer"], "features.8")
EXPLAIN_RESOLUTION = int(get_yaml_val(["explainability", "resolution"], 300))

# Security and uploads
UPLOAD_MAX_SIZE_MB = int(get_yaml_val(["security", "upload_max_size_mb"], 10))
ALLOWED_EXTENSIONS = set(get_yaml_val(["security", "allowed_extensions"], ["jpg", "jpeg", "png"]))
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwtsecretkeywhichis32byteslong")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
