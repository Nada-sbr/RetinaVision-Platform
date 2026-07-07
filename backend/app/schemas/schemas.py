import uuid
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ----------------------------------------------------
# Token & Auth Schemas
# ----------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


# ----------------------------------------------------
# User Schemas
# ----------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
    role: Optional[str] = "ophthalmologist"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------------------
# Patient Schemas
# ----------------------------------------------------
class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    birth_date: date
    gender: str = Field(..., description="Male, Female, or Other")


class PatientOut(BaseModel):
    id: int
    first_name: str
    last_name: str
    birth_date: date
    gender: str
    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------------------
# Prediction Schemas
# ----------------------------------------------------
class PredictionOut(BaseModel):
    id: int
    image_id: uuid.UUID
    n_prob: float
    d_prob: float
    g_prob: float
    c_prob: float
    a_prob: float
    h_prob: float
    m_prob: float
    o_prob: float
    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------------------
# Report Schemas
# ----------------------------------------------------
class ReportOut(BaseModel):
    id: int
    image_id: uuid.UUID
    report_text: str
    status: str
    validated_by: Optional[int] = None
    validated_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ----------------------------------------------------
# Image & History Schemas
# ----------------------------------------------------
class ImageOut(BaseModel):
    id: uuid.UUID
    patient_id: int
    eye: str
    original_filename: str
    filepath: str
    gradcam_filepath: Optional[str] = None
    uploaded_by: int
    uploaded_at: datetime

    # Nested relations if present
    prediction: Optional[PredictionOut] = None
    report: Optional[ReportOut] = None

    class Config:
        from_attributes = True


# Combined History Response
class HistoryItem(BaseModel):
    image: ImageOut
    patient: PatientOut
    prediction: Optional[PredictionOut] = None
    report: Optional[ReportOut] = None

    class Config:
        from_attributes = True
