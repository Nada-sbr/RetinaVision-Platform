# Architecture Specification

This document details the software design, database schemas, and ML pipeline structures of the Ocular AI platform.

---

## 🏛️ System Component Breakdown

1. **Frontend (Nginx + React)**:
   - Provides a light, clean, clinical user interface.
   - Reverse-proxies API calls `/api/*` to the FastAPI backend container to solve CORS issues.
2. **Backend (FastAPI)**:
   - Coordinates Python application logic.
   - Dynamically loads model weights and executes PyTorch predictions and OpenCV overlays.
3. **Database (PostgreSQL 15)**:
   - Securely stores patient files, credentials, logs, and diagnostic history.
4. **LLM Server (Ollama Gemma:2b)**:
   - Runs locally on the host machine to generate structured clinical reports.

---

## 🗄️ Database Schema ERD

```
  +------------------+         +------------------+
  |      users       |         |     patients     |
  +------------------+         +------------------+
  | id (PK)          |         | id (PK)          |
  | email            |         | first_name       |
  | hashed_password  |         | last_name        |
  | full_name        |         | birth_date       |
  | role             |         | gender           |
  +--------+---------+         +--------+---------+
           |                            |
           | has uploaded               | registered to
           v                            v
  +-----------------------------------------------+
  |                     images                    |
  +-----------------------------------------------+
  | id (PK)                                       |
  | patient_id (FK)                               |
  | eye (left / right)                            |
  | original_filename                             |
  | filepath                                      |
  | uploaded_by (FK)                              |
  +--------+----------------------------+---------+
           |                            |
           | has prediction             | has report
           v                            v
  +------------------+         +------------------+
  |   predictions    |         |     reports      |
  +------------------+         +------------------+
  | image_id (PK, FK)|         | image_id (PK, FK)|
  | n_prob (Normal)  |         | report_text      |
  | d_prob (Diabetes)|         | created_at       |
  | g_prob (Glaucoma)|         +------------------+
  | c_prob (Cataract)|
  | a_prob (AMD)     |
  | h_prob (Hyper)   |
  | m_prob (Myopia)  |
  | o_prob (Other)   |
  +------------------+
```

---

## 🧠 Retinal Diagnostic CNN Pipeline

```
Retinal JPEG File
       ↓
Preprocessed (Resize 300x300, Normalized using ImageNet stats)
       ↓
EfficientNet-B3 Backbone
       ├─→ Features stages 7 & 8 (Trainable/Fine-tuning)
       ├─→ Final Convolutional Layer (Target for Grad-CAM activations)
       ↓
Linear Classifier (8 disease logits outputs)
       ├─→ Sigmoid (Probability mapped values)
       └─→ Focal Loss calculation (Weights handling imbalance)
```
- **Image Size**: $300\times300$ pixels (Native resolution of EfficientNet-B3).
- **Explainability Layer**: Grad-CAM captures hook gradients on stage `features.8` of the EfficientNet structure to outline active spatial heatmaps.
- **Reporting Prompts**: Synthesizes the computed prediction probabilities, eye context, and patient age, formatting them into Ollama instructions for report outputs.
