from contextlib import asynccontextmanager
from fastapi import FastAPI
from detection.engine.detector import FaceDetector
from app.core.config import settings

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Initializing MediaPipe Face Detector (Confidence: {settings.MIN_DETECTION_CONFIDENCE})...")
    # Inject the setting here
    ml_models["face_detector"] = FaceDetector(min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE)
    
    yield
    
    print("Shutting down Face Detector and clearing memory...")
    if "face_detector" in ml_models:
        ml_models["face_detector"].close()
    ml_models.clear()