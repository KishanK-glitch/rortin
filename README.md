
# Face Detection API & Web Interface

A lightweight, decoupled Face Detection system. It features a high-performance REST API built with FastAPI and a clean testing interface built with Streamlit. 

The machine learning engine runs on OpenCV's Deep Neural Network (DNN) module using a ResNet model. This bypasses the heavy dependencies, C-binding conflicts, and massive Docker footprint associated with frameworks like YOLO or modern MediaPipe, making it highly stable for CPU deployment.

## Project Structure

```text
face_detection/
├── .github/workflows/       # Automated CI/CD
├── app/
│   ├── api/
│   │   ├── dependencies.py  # Model injection
│   │   └── routes.py        # API endpoints
│   ├── core/
│   │   ├── config.py        # Environment variables
│   │   └── lifespan.py      # Memory management & startup
│   └── main.py              # FastAPI application entry
├── detection/
│   ├── engine/
│   │   ├── detector.py      # OpenCV DNN inference logic
│   │   ├── preprocessor.py  # Bytes to BGR image conversion
│   │   └── postprocessor.py # Matrix output to clean JSON
├── frontend/
│   └── app.py               # Streamlit web interface
├── .gitignore
├── requirements.txt
└── README.md

```

## Prerequisites

* Python 3.9+
* A standard Python Virtual Environment (`venv`) to isolate dependencies.

## Installation & Setup

**1. Clone the repository and navigate into it:**

```bash
git clone [https://github.com/KishanK-glitch/rortin.git](https://github.com/KishanK-glitch/rortin.git)
cd rortin

```

**2. Create and activate a virtual environment:**

```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
pip install streamlit requests pillow

```

**4. Download the ML Model Weights:**
The OpenCV DNN engine requires specific pre-trained Caffe models. Run these commands from the root directory to download them:

*(Windows PowerShell)*

```powershell
Invoke-WebRequest -Uri "[https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)" -OutFile "deploy.prototxt"
Invoke-WebRequest -Uri "[https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel)" -OutFile "res10_300x300_ssd_iter_140000.caffemodel"

```

*(Linux/macOS bash)*

```bash
wget [https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
wget [https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel)

```

## Running the Application

This project runs as two decoupled services. You will need two separate terminal windows.

### Terminal 1: Boot the API Server

Ensure your virtual environment is active, then start the FastAPI backend:

```bash
uvicorn app.main:app --reload

```

* The API runs at: `http://127.0.0.1:8000`
* Swagger UI Documentation: `http://127.0.0.1:8000/docs`

### Terminal 2: Boot the Frontend

Open a new terminal, activate the virtual environment, and start the Streamlit UI:

```bash
streamlit run frontend/app.py

```

* The Web UI runs at: `http://localhost:8501`

## API Reference

### `POST /api/detect`

Accepts a standard multipart/form-data image upload and returns a JSON payload with absolute pixel coordinates for bounding boxes.

**Request:**

* `file`: (Image binary)

**Response (200 OK):**

```json
{
  "filename": "image.jpg",
  "face_count": 1,
  "faces": [
    {
      "confidence": 0.9998,
      "bounding_box": {
        "x_min": 107,
        "y_min": 34,
        "width": 63,
        "height": 89
      }
    }
  ]
}

```

```

```
