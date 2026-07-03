from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.database import Base, engine
from app.api.endpoints import router
from app.api.health import router as health_router
from app.core.logging_config import setup_logging

# Setup logging system
setup_logging()

# Automatically create tables in local DB on startup if they don't exist
# This is highly convenient for development.
Base.metadata.create_all(bind=engine)

from app.core import config

app = FastAPI(
    title=config.PROJECT_NAME,
    description="Backend API for Fundus Image Disease Classification, Grad-CAM overlays, and LLM clinical reporting.",
    version=config.VERSION
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve original images and Grad-CAM overlays
# Path is: backend/static
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include the routers
app.include_router(router)
app.include_router(health_router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_docs_url": "/docs",
        "health_check": "passed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
