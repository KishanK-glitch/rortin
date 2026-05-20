from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from detection.engine.detector import FaceDetector
from detection.engine.preprocessor import prepare_image
from detection.engine.postprocessor import format_results
from app.api.dependencies import get_face_detector

router = APIRouter()

@router.post("/detect")
async def detect_faces(
    file: UploadFile = File(...),
    detector: FaceDetector = Depends(get_face_detector)
):
    # Fail fast if the upload isn't an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read file bytes into memory
        image_bytes = await file.read()
        
        # 1. Preprocess: Bytes -> RGB Array
        img_rgb, height, width = prepare_image(image_bytes)
        
        # 2. Inference: Run MediaPipe
        results = detector.process(img_rgb)
        
        # 3. Postprocess: Normalized Coordinates -> Pixel Values
        faces = format_results(results, height, width)
        
        return {
            "filename": file.filename,
            "face_count": len(faces),
            "faces": faces
        }

    except ValueError as e:
        # Caught from our preprocessor if cv2 fails to decode
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected ML engine failures
        raise HTTPException(status_code=500, detail="Internal server error during face detection.")