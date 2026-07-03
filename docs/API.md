# API Endpoint Reference Documentation

This document lists all active endpoints exposed by the Ocular AI FastAPI backend router.

## 🔑 Authentication Endpoints

### 1. `POST /api/register`
Creates a new clinical user (Ophthalmologist/Researcher).
- **Request Body**:
  ```json
  {
    "email": "doctor@test.com",
    "password": "securepassword123",
    "full_name": "Dr. Test User",
    "role": "ophthalmologist"
  }
  ```
- **Response**: `201 Created` returning the user model without the password.

### 2. `POST /api/login`
Obtains a bearer access token.
- **Form Data**:
  - `username`: Email address
  - `password`: Password
- **Response**: `200 OK`
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "token_type": "bearer"
  }
  ```

---

## 📁 Patient and Image Management

### 3. `POST /api/patients`
Registers a new patient. (Requires Authentication Header).
- **Request Body**:
  ```json
  {
    "first_name": "Jane",
    "last_name": "Doe",
    "birth_date": "1985-04-12",
    "gender": "Female"
  }
  ```
- **Response**: `201 Created`

### 4. `POST /api/upload-image`
Uploads a fundus retina photograph. (Requires Authentication Header).
- **Multipart Form Data**:
  - `patient_id`: UUID
  - `eye`: "left" or "right"
  - `file`: Image binary (JPEG/PNG)
- **Response**: `201 Created` containing the generated image ID.

---

## 🧠 Model Predictions & Explanations

### 5. `POST /api/predict/{image_id}`
Triggers EfficientNet-B3 model inference on the uploaded image. (Requires Authentication Header).
- **Response**: `200 OK` with probabilities for the 8 labels:
  ```json
  {
    "image_id": "UUID",
    "n_prob": 0.05,
    "d_prob": 0.85,
    "g_prob": 0.02,
    "c_prob": 0.01,
    "a_prob": 0.01,
    "h_prob": 0.04,
    "m_prob": 0.01,
    "o_prob": 0.01
  }
  ```

### 6. `POST /api/explain/{image_id}`
Runs the Grad-CAM backward propagation layer to highlight feature activations. (Requires Authentication Header).
- **Response**: `200 OK` returning paths to original and colored overlay images.

### 7. `POST /api/report/{image_id}`
Triggers Ollama synthesis to write the clinical report. (Requires Authentication Header).
- **Response**: `200 OK` with the synthesised text.

---

## 🩺 Production Monitoring Endpoints

### 8. `GET /health`
Standard live status query.
- **Response**: `{"status": "healthy", "timestamp": "..."}`

### 9. `GET /ready`
Readiness status checking DB and model file locks.
- **Response**: `200 OK` or `503 Service Unavailable`.

### 10. `GET /metrics`
Exposes uptime, memory usage, database row count, and average prediction latency.
- **Response**: `200 OK`.

### 11. `GET /monitoring/drift-report`
Returns Evidently AI data/prediction drift audit dashboard.
- **Response**: `200 OK` with Content-Type `text/html`.
