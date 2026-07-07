from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Date, Boolean, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, date
from typing import List, Optional

from app.core.database import Base


class User(Base):
    """
    User model representing clinicians, ophtalmologists, or admins.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="ophthalmologist")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="uploader")
    reports: Mapped[List["Report"]] = relationship("Report", back_populates="validator")


class Patient(Base):
    """
    Patient model representing individuals undergoing ocular diagnostic imaging.
    """

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[str] = mapped_column(String, nullable=False)
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="patient", cascade="all, delete-orphan")


class Image(Base):
    """
    Image model representing uploaded fundus images.
    """

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    eye: Mapped[str] = mapped_column(String, nullable=False)  # "left" or "right"
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)  # Path to saved original image on disk
    gradcam_filepath: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # Path to generated Grad-CAM overlay
    uploaded_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="images")
    uploader: Mapped["User"] = relationship("User", back_populates="images")
    prediction: Mapped["Prediction"] = relationship(
        "Prediction", back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
    report: Mapped["Report"] = relationship("Report", back_populates="image", uselist=False, cascade="all, delete-orphan")


class Prediction(Base):
    """
    Prediction model representing binary indicators and probabilities for all 8 ODIR labels.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Probabilities for the 8 labels (N, D, G, C, A, H, M, O)
    n_prob: Mapped[float] = mapped_column(Float, nullable=False)
    d_prob: Mapped[float] = mapped_column(Float, nullable=False)
    g_prob: Mapped[float] = mapped_column(Float, nullable=False)
    c_prob: Mapped[float] = mapped_column(Float, nullable=False)
    a_prob: Mapped[float] = mapped_column(Float, nullable=False)
    h_prob: Mapped[float] = mapped_column(Float, nullable=False)
    m_prob: Mapped[float] = mapped_column(Float, nullable=False)
    o_prob: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    image: Mapped["Image"] = relationship("Image", back_populates="prediction")


class Report(Base):
    """
    Report model representing generated clinical LLM reports.
    """

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, default="draft")  # "draft" or "validated"
    validated_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    image: Mapped["Image"] = relationship("Image", back_populates="report")
    validator: Mapped["User"] = relationship("User", back_populates="reports")
