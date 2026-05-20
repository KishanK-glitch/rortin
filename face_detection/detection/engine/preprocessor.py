import cv2
import numpy as np

def prepare_image(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
    """
    Converts raw image bytes from an API upload into an RGB numpy array.
    Returns the RGB image and its original dimensions (height, width).
    """
    # 1. Convert bytes to a 1D numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. Decode the array into an OpenCV image (which defaults to BGR)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Invalid image file. Could not decode bytes.")

    # Get dimensions before conversion
    height, width, _ = img_bgr.shape

    # 3. MediaPipe requires RGB format, so convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, height, width