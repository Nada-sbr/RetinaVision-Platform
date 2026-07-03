import pytest
import sys
import os
from datetime import date
import uuid

# Add backend app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.models import Patient, Image, Prediction, User

def test_database_crud(db):
    """Test user, patient, image, and prediction storage and retrieval."""
    # 1. Create a user
    user = User(
        email="doctor_db_test@test.com",
        hashed_password="fakehashpwd123",
        full_name="Dr. DB Test",
        role="ophthalmologist"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    assert user.id is not None
    
    # 2. Create a patient
    patient = Patient(
        first_name="Alice",
        last_name="Smith",
        birth_date=date(1990, 5, 15),
        gender="Female"
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    assert patient.id is not None
    
    # 3. Create an image
    image = Image(
        id=uuid.uuid4(),
        patient_id=patient.id,
        eye="left",
        original_filename="left_eye.jpg",
        filepath="static/uploads/original/test.jpg",
        uploaded_by=user.id
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    assert image.id is not None
    
    # 4. Create predictions
    prediction = Prediction(
        image_id=image.id,
        n_prob=0.9,
        d_prob=0.1,
        g_prob=0.0,
        c_prob=0.0,
        a_prob=0.0,
        h_prob=0.0,
        m_prob=0.0,
        o_prob=0.0
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    assert prediction.image_id == image.id
    assert prediction.n_prob == 0.9
    
    # Verify cascade delete
    db.delete(patient)
    db.commit()
    
    # Image and Prediction should be deleted automatically by cascade
    assert db.query(Image).filter(Image.id == image.id).first() is None
    assert db.query(Prediction).filter(Prediction.image_id == image.id).first() is None
