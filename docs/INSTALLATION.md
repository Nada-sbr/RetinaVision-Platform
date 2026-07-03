# Local Installation Guide

This document describes how to set up the Ocular AI project locally on your development machine for development and testing.

## Prerequisites

- **Python**: Version `3.12` installed.
- **Node.js**: Version `18+` and `npm` installed.
- **Git**: Installed.
- **Ollama**: Installed locally on the host machine.

---

## 1. Local Environment Configuration

Copy the `.env.example` file to `.env` and fill in the values:
```bash
cp .env.example .env
```
Ensure that `DATABASE_URL` is set to SQLite for local development:
```env
DATABASE_URL=sqlite:///./test.db
OLLAMA_URL=http://localhost:11434
MODEL_PATH=checkpoints/best_model.pth
JWT_SECRET=supersecretjwtsecretkeywhichis32byteslong
ACCESS_TOKEN_EXPIRE_MINUTES=60
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

---

## 2. Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Alembic migrations:
   ```bash
   python -m alembic upgrade head
   ```
5. Start the FastAPI development server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
The API Swagger documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 3. Frontend Setup (React/Vite)

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
The frontend application will be running at [http://127.0.0.1:5173](http://127.0.0.1:5173).

---

## 4. Local Ollama Configuration

Ocular AI uses the local Ollama API to run LLM reports.
1. Download Ollama from [https://ollama.com](https://ollama.com).
2. Download the Gemma 2B model:
   ```bash
   ollama pull gemma:2b
   ```
3. Verify it is running by sending a request or starting the interactive session:
   ```bash
   ollama run gemma:2b
   ```
Ensure Ollama is running on port `11434` (`http://localhost:11434`).
