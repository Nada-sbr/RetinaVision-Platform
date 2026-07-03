#  Ocular AI — Enterprise Retinal Disease Diagnostic & MLOps Platform

[![Build Status](https://github.com/your-username/ocular-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ocular-ai/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/data-versioned%20with%20DVC-green.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/experiments-tracked%20in%20MLflow-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Ocular AI is an enterprise-grade clinical decision support system (CDSS) for retinal disease detection using deep learning (EfficientNet-B3), visual explainability (Grad-CAM), and LLM clinical reporting (Ollama Gemma:2b). The application is fully containerized, versioned, tracked, and monitored using modern MLOps best practices (DVC, MLflow, Evidently AI, PyTest, GitHub Actions).

---

##  System Architecture

```
                       +-------------------+
                       |      Browser      |
                       +-------------------+
                                 ^
                                 | HTTP / WebSocket
                                 v
                       +-------------------+
                       |    Nginx Proxy    |
                       +-------------------+
                                 ^
                                 | Proxy Pass
                                 v
                       +-------------------+
                       |  FastAPI Backend  |
                       +-------------------+
                        /        |        \
                       /         |         \
                      v          v          v
          +------------+   +-----------+   +-------------+
          | EfficientNet|   | PostgreSQL|   | Ollama LLM  |
          |   Grad-CAM |   |   (DB)    |   | (Gemma:2b)  |
          +------------+   +-----------+   +-------------+
                 ^                               ^
                 | Track Experiments             | Generate Reports
                 v                               v
          +------------+                   +-------------+
          |   MLflow   |                   |  Evidently  |
          |  Registry  |                   |  Drift API  |
          +------------+                   +-------------+
```

---

##  Key Engineering Features

- **Model Architecture**: Fine-tuned **EfficientNet-B3** (300x300 native input resolution) with custom Multi-Label Focal Loss handling severe class imbalances across 8 classes (Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other).
- **Explainable AI (XAI)**: Heatmap localization using **Grad-CAM** hooked onto the final feature stage (`features.8`). Spatial projections identify the target pathology zone textually.
- **Experiment Tracking (MLflow)**: Full logging of epochs, learning rates, loss functions, batch sizes, confusion matrices, and final metrics (subset accuracy, macro F1, macro ROC-AUC). Automated registration in Model Registry.
- **Data & Model Versioning (DVC)**: Fully reproducible pipeline mapping raw and preprocessed datasets to model checkpoints.
- **Continuous Monitoring**: Automatic prediction drift and data drift analysis using **Evidently AI** with direct HTML report downloads.
- **CI/CD Pipeline**: Pre-commit hooks and GitHub Actions workflows checking coding standards (Black, Ruff, Flake8), running unit tests (PyTest with mocked services), and compiling Docker images.

---

##  Technology Stack

- **ML Frameworks**: PyTorch, Torchvision, OpenCV, Albumentations.
- **Ops & DevOps**: MLflow, DVC, Evidently AI, Docker, Docker Compose, Nginx.
- **Backend API**: FastAPI, Uvicorn, SQLAlchemy 2.0 (PostgreSQL/SQLite), Alembic, Pydantic v2.
- **Frontend**: React, Vite, CSS custom light theme (clinical white aesthetics).

---

##  Quick Start (Docker Compose)

### Prerequisites
- Docker and Docker Compose installed.
- Ollama installed on your host system running `gemma:2b` (`ollama run gemma:2b`).

### Deployment
To start all services, run the following from the project root:
```bash
docker-compose up --build -d
```
This launches 4 orchestrated containers:
1. **ocular-postgres**: Database engine on port `5432`.
2. **ocular-backend**: FastAPI server on port `8000` (auto-applies Alembic migrations).
3. **ocular-frontend**: Nginx reverse proxy on port `80` serving the React client.
4. **ocular-mlflow**: MLflow tracking server on port `5000`.

Open your browser at [http://localhost](http://localhost) to access the platform, and [http://localhost:5000](http://localhost:5000) to view the MLflow UI.

---

##  Platform Documentation

Detailed architecture and MLOps deployment documentation can be found in the `docs/` folder:

   [Installation Guide](file:///c:/Users/Lenovo/Desktop/Ocular_Diseases%20Project/Project/docs/INSTALLATION.md) — Manual setups for local development.
   [Deployment Guide](file:///c:/Users/Lenovo/Desktop/Ocular_Diseases%20Project/Project/docs/DEPLOYMENT.md) — Production Docker deployments.
   [API Documentation](file:///c:/Users/Lenovo/Desktop/Ocular_Diseases%20Project/Project/docs/API.md) — Router schema definitions and endpoints.
   [MLOps Guide](file:///c:/Users/Lenovo/Desktop/Ocular_Diseases%20Project/Project/docs/MLOPS.md) — MLflow server and DVC pipeline tracking.
   [Architecture Specification](file:///c:/Users/Lenovo/Desktop/Ocular_Diseases%20Project/Project/docs/ARCHITECTURE.md) — System designs, ML structures, and schema tables.

---
##  Testing

We use **PyTest** to run all unit and integration checks. Disabling capture prevents Windows DLL conflicts:
```bash
cd backend
python -m pytest -s tests/
```

---

##  Clinical Disclaimer

Ocular AI is a clinical decision support system (CDSS) designed for medical screening research. The predictions and reports generated by the CNN and LLM pipelines do not constitute a definitive medical diagnosis. Therapeutic decisions and clinical validations remain the exclusive responsibility of the licensed ophthalmologist.
