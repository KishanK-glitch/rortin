import numpy as np

def format_results(dnn_results, image_height: int, image_width: int) -> list[dict]:
    faces = []
    
    detections = dnn_results[0, 0]
    
    for i in range(detections.shape[0]):
        confidence = float(detections[i, 2])
        
        if confidence >= 0.5:
            box = detections[i, 3:7] * np.array([image_width, image_height, image_width, image_height])
            xmin, ymin, xmax, ymax = box.astype("int")
            
            faces.append({
                "confidence": round(confidence, 4),
                "bounding_box": {
                    # Explicitly cast to native Python integers
                    "x_min": int(max(0, xmin)),
                    "y_min": int(max(0, ymin)),
                    "width": int(max(0, xmax - xmin)),
                    "height": int(max(0, ymax - ymin))
                }
            })

    return faces