from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import os
import sys
import uuid
import shutil
from datetime import datetime, date
from typing import List
from PIL import Image
import numpy as np
import torch
import cv2
import logging
import time

logger = logging.getLogger(__name__)

# Ensure project root is in the path to import ml_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token, get_current_user
from app.models import models
from app.schemas import schemas

# Import ML pipeline modules
from ml_pipeline.model_effnet import ODIREfficientNetB3
from ml_pipeline.gradcam import GradCAM
from ml_pipeline.transforms import get_val_transforms
from ml_pipeline.report_generator import ClinicalReportGenerator
from ml_pipeline.llm_service import LLMService

router = APIRouter(prefix="/api")

# Directories for uploads
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static", "uploads"))
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
GRADCAM_DIR = os.path.join(UPLOAD_DIR, "gradcam")
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ----------------------------------------------------
# Global ML/LLM Inferences Configurations
# ----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from app.core import config

# Find the model checkpoints path dynamically for both Docker container and local development
MODEL_PATH = os.path.abspath(config.MODEL_PATH)
OLLAMA_URL = config.OLLAMA_URL

# Helper to load PyTorch model once or on-demand
_model_instance = None
def get_prediction_model():
    global _model_instance
    if _model_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Trained model checkpoint not found at {MODEL_PATH}. Please train the model first.")
        model = ODIREfficientNetB3(num_classes=8, pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _model_instance = model
    return _model_instance

# ----------------------------------------------------
# AUTHENTICATION ENDPOINTS
# ----------------------------------------------------
@router.post("/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user_in.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_pwd = get_password_hash(user_in.password)
    user = models.User(
        email=user_in.email,
        hashed_password=hashed_pwd,
        full_name=user_in.full_name,
        role=user_in.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# ----------------------------------------------------
# PATIENT REGISTRY ENDPOINTS
# ----------------------------------------------------
@router.post("/patients", response_model=schemas.PatientOut, status_code=status.HTTP_201_CREATED)
def create_patient(patient_in: schemas.PatientCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    patient = models.Patient(
        first_name=patient_in.first_name,
        last_name=patient_in.last_name,
        birth_date=patient_in.birth_date,
        gender=patient_in.gender
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@router.get("/patients/{patient_id}", response_model=schemas.PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

# ----------------------------------------------------
# IMAGE UPLOAD ENDPOINT
# ----------------------------------------------------
@router.post("/upload-image", response_model=schemas.ImageOut, status_code=status.HTTP_201_CREATED)
def upload_image(
    patient_id: int = Form(...),
    eye: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Verify patient exists
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    if eye.lower() not in ["left", "right"]:
        raise HTTPException(status_code=400, detail="Eye value must be either 'left' or 'right'")
        
    # Enforce file format security checks
    file_ext = os.path.splitext(file.filename)[1].lstrip('.').lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Unsupported file format. Allowed formats: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )
        
    # Enforce file size security checks
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0) # Reset pointer
    
    max_size_bytes = config.UPLOAD_MAX_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {config.UPLOAD_MAX_SIZE_MB}MB."
        )

    # Generate unique filename on disk
    db_img_id = uuid.uuid4()
    filename_on_disk = f"{db_img_id}.{file_ext}"
    filepath_on_disk = os.path.join(ORIGINAL_DIR, filename_on_disk)
    
    # Save original image to disk
    with open(filepath_on_disk, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Relative path for frontend serving (CORS/static mount)
    relative_filepath = f"static/uploads/original/{filename_on_disk}"
    
    image = models.Image(
        id=db_img_id,
        patient_id=patient_id,
        eye=eye.lower(),
        original_filename=file.filename,
        filepath=relative_filepath,
        uploaded_by=current_user.id
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image

# ----------------------------------------------------
# DEEP LEARNING PREDICTION ENDPOINT
# ----------------------------------------------------
@router.post("/predict/{image_id}", response_model=schemas.PredictionOut)
def predict_image(image_id: uuid.UUID, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # 1. Fetch image
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Check if prediction already exists
    if image.prediction:
        return image.prediction
        
    # 2. Get full path of image on disk
    full_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", image.filepath))
    if not os.path.exists(full_img_path):
        raise HTTPException(status_code=404, detail="Image file missing on disk server")
        
    # 3. Load model and run inference
    logger.info(f"Running inference for image {image_id}...")
    start_time = time.time()
    try:
        model = get_prediction_model()
        pil_img = Image.open(full_img_path).convert('RGB')
        val_transforms = get_val_transforms(300)
        input_tensor = val_transforms(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        duration = time.time() - start_time
        logger.info(f"Inference succeeded for image {image_id} in {duration:.4f}s. Raw output: {probs}")
        
        # Import dynamically to avoid circular import issues
        from app.api.health import log_inference_time
        log_inference_time(duration)
    except Exception as e:
        logger.error(f"Inference failed for image {image_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
    # 4. Save predictions in DB
    prediction = models.Prediction(
        image_id=image_id,
        n_prob=float(probs[0]),
        d_prob=float(probs[1]),
        g_prob=float(probs[2]),
        c_prob=float(probs[3]),
        a_prob=float(probs[4]),
        h_prob=float(probs[5]),
        m_prob=float(probs[6]),
        o_prob=float(probs[7])
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

# ----------------------------------------------------
# EXPLAINABLE AI (GRAD-CAM) ENDPOINT
# ----------------------------------------------------
@router.post("/explain/{image_id}", response_model=schemas.ImageOut)
def explain_image(image_id: uuid.UUID, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    if not image.prediction:
        raise HTTPException(status_code=400, detail="Please run predictions first via POST /api/predict/{image_id}")
        
    # Paths
    full_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", image.filepath))
    
    # 1. Run hook-based Grad-CAM
    try:
        model = get_prediction_model()
        target_layer = model.backbone.features[8]
        gcam = GradCAM(model, target_layer)
        
        # Load image and process
        pil_img = Image.open(full_img_path).convert('RGB')
        original_cv2 = np.array(pil_img)
        val_transforms = get_val_transforms(300)
        input_tensor = val_transforms(pil_img).unsqueeze(0).to(DEVICE)
        
        # Select target class (class with highest predicted probability)
        pred_probs = [
            image.prediction.n_prob, image.prediction.d_prob, image.prediction.g_prob, image.prediction.c_prob,
            image.prediction.a_prob, image.prediction.h_prob, image.prediction.m_prob, image.prediction.o_prob
        ]
        class_idx = int(np.argmax(pred_probs))
        
        # Generate raw heatmap
        heatmap = gcam.generate_heatmap(input_tensor, class_idx)
        
        # Overlay heatmap on original image
        _, overlayed_img = gcam.overlay_heatmap(heatmap, original_cv2, alpha=0.4)
        
        # Save overlay image to disk
        gcam_filename = f"gradcam_{image_id}.png"
        gcam_filepath_on_disk = os.path.join(GRADCAM_DIR, gcam_filename)
        cv2.imwrite(gcam_filepath_on_disk, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
        
        # Update database path
        relative_gcam_path = f"static/uploads/gradcam/{gcam_filename}"
        image.gradcam_filepath = relative_gcam_path
        db.commit()
        db.refresh(image)
        
        # Remove handles to prevent memory leaks
        gcam.remove_handles()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")
        
    return image

# ----------------------------------------------------
# CLINICAL REPORT GENERATION ENDPOINT (MISTRAL LLM)
# ----------------------------------------------------
@router.post("/report/{image_id}", response_model=schemas.ReportOut)
def generate_clinical_report(image_id: uuid.UUID, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    if not image.prediction:
        raise HTTPException(status_code=400, detail="Please run predictions first via POST /api/predict/{image_id}")
        
    # Check if report already exists
    if image.report:
        return image.report
        
    # Load raw heatmap for spatial localization
    full_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", image.filepath))
    try:
        model = get_prediction_model()
        target_layer = model.backbone.features[8]
        gcam = GradCAM(model, target_layer)
        pil_img = Image.open(full_img_path).convert('RGB')
        val_transforms = get_val_transforms(300)
        input_tensor = val_transforms(pil_img).unsqueeze(0).to(DEVICE)
        
        pred_probs = [
            image.prediction.n_prob, image.prediction.d_prob, image.prediction.g_prob, image.prediction.c_prob,
            image.prediction.a_prob, image.prediction.h_prob, image.prediction.m_prob, image.prediction.o_prob
        ]
        class_idx = int(np.argmax(pred_probs))
        heatmap = gcam.generate_heatmap(input_tensor, class_idx)
        gcam.remove_handles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap extraction failed for report: {str(e)}")
        
    # 1. Compile payload
    labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    predictions_dict = {l: float(p) for l, p in zip(labels, pred_probs)}
    
    patient = image.patient
    # Calculate age from birth_date
    today = date.today()
    age = today.year - patient.birth_date.year - ((today.month, today.day) < (patient.birth_date.month, patient.birth_date.day))
    
    report_gen = ClinicalReportGenerator()
    payload = report_gen.generate_json_payload(age, patient.gender, predictions_dict, heatmap)
    
    # 2. Build prompt
    prompt = report_gen.build_llm_prompt(payload)
    
    # 3. Request LLM Inference (Mistral)
    try:
        llm = LLMService(ollama_url=OLLAMA_URL)
        report_text = llm.generate_report(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Service execution failed: {str(e)}")
        
    # 4. Save Report in DB
    report = models.Report(
        image_id=image_id,
        report_text=report_text,
        status="draft"
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

# ----------------------------------------------------
# CLINICAL HISTORY QUERY ENDPOINT
# ----------------------------------------------------
@router.get("/history", response_model=List[schemas.HistoryItem])
def get_clinical_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Retrieve all images with loaded relations
    images = db.query(models.Image).order_by(models.Image.uploaded_at.desc()).all()
    
    history = []
    for img in images:
        history.append({
            "image": img,
            "patient": img.patient,
            "prediction": img.prediction,
            "report": img.report
        })
    return history
