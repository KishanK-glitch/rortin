import cv2
import numpy as np

class FaceDetector:
    def __init__(self, prototxt="deploy.prototxt", model="res10_300x300_ssd_iter_140000.caffemodel", min_detection_confidence=0.5):
        # Loads the neural network directly into OpenCV
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.threshold = min_detection_confidence

    def process(self, image_rgb: np.ndarray):
        # OpenCV DNN expects BGR images. Convert RGB back to BGR.
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Resize to 300x300 and normalize as required by this specific model
        blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Run inference
        self.net.setInput(blob)
        return self.net.forward()

    def close(self):
        # OpenCV handles memory automatically, no explicit close needed
        pass