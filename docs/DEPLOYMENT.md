# Production Deployment Guide

This document describes how to deploy the Ocular AI platform in production using Docker Compose.

## 🐳 Container Architecture

The production environment consists of 4 orchestrated services:
- **PostgreSQL**: Relational database storage.
- **FastAPI Backend**: Machine learning model inference and clinical report synthesis.
- **Nginx Proxy / React**: Frontend web portal serving.
- **MLflow Tracking**: Experiment and model logging server.

---

## 1. Setup secrets in `.env`

Create a `.env` file in the root directory (never commit this file to git):
```env
DATABASE_URL=postgresql://postgres:secure_production_password@db:5432/ocular_db
OLLAMA_URL=http://host.docker.internal:11434
MODEL_PATH=checkpoints/best_model.pth
JWT_SECRET=YOUR_SECURE_JWT_SECRET_32_CHARS_LONG
ACCESS_TOKEN_EXPIRE_MINUTES=60
ENVIRONMENT=production
CORS_ORIGINS=http://yourdomain.com
```

---

## 2. Docker Compose Deploy

Build and start the application in detached mode:
```bash
docker-compose up --build -d
```

Verify that all containers are healthy:
```bash
docker-compose ps
```

The services will be exposed as follows:
- **Web App UI (Nginx)**: Port `80` (accessible via `http://localhost`)
- **FastAPI Backend**: Port `8000` (accessible via `http://localhost:8000/docs`)
- **MLflow Tracking Server**: Port `5000` (accessible via `http://localhost:5000`)
- **PostgreSQL Database**: Port `5432` (internal network binding only, or external access if allowed)

---

## 3. Data Volumes and Persistence

Docker Compose maps persistent volumes to ensure data is preserved across service restarts:
- `postgres_data` -> Maps to `/var/lib/postgresql/data` (stores all SQL patients, credentials, predictions, and reports).
- `backend_uploads` -> Maps to `/app/static/uploads` (stores uploaded original fundus files and Grad-CAM output heatmaps).
- `mlflow_data` -> Maps to `/mlflow` (stores local tracking runs and logged PyTorch model registry weights).

---

## 4. Production Security Rules

1. **Change default credentials**: Change PostgreSQL username/password from `postgres/postgres` in the environment variables before deploying to staging/prod.
2. **CORS strict routing**: Enter your production domain URL in `CORS_ORIGINS` to prevent unauthorized domain requests.
3. **Volume backups**: Set up backups for `postgres_data` and `backend_uploads` volumes.
