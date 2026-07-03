FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies required for OpenCV and PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and Torchvision CPU edition first from the PyTorch repository
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu

# Copy and install the rest of the Python dependencies
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML pipeline and checkpoints
COPY ml_pipeline /app/ml_pipeline
COPY checkpoints /app/checkpoints

# Copy backend app source code and Alembic migrations configuration
COPY backend/app /app/app
COPY backend/alembic.ini /app/alembic.ini
COPY backend/alembic /app/alembic

# Create static directories for uploads
RUN mkdir -p /app/static/uploads/original /app/static/uploads/gradcam /app/logs

# Add non-root system user for security
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check to ensure containers are monitored in production
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Script to run migrations and start FastAPI server
CMD ["sh", "-c", "python -m alembic upgrade head || true && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
