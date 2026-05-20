import os

class Settings:
    PROJECT_NAME: str = "Face Detection API"
    VERSION: str = "1.0.0"
    
    # ML Model Configuration
    # Defaults to 0.5 if not set in a .env file or system environment
    MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))

settings = Settings()