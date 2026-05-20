from fastapi import HTTPException
from app.core.lifespan import ml_models
from detection.engine.detector import FaceDetector

def get_face_detector() -> FaceDetector:
    """Injects the pre-loaded MediaPipe model into API routes."""
    detector = ml_models.get("face_detector")
    if not detector:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")
    return detector