import streamlit as st
import requests
from PIL import Image, ImageDraw

# Point this to your FastAPI server
API_URL = "http://127.0.0.1:8000/api/detect"

st.set_page_config(page_title="Face Detection", layout="centered")
st.title("Face Detection Engine")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the raw image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Processing..."):
            # Reset file pointer before reading
            uploaded_file.seek(0)
            
            # Format the payload for FastAPI's UploadFile
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Fire the POST request to the API
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                result = response.json()
                
                faces = result.get("faces", [])
                st.success(f"Found {len(faces)} face(s).")
                
                # Draw the bounding boxes on the image
                draw = ImageDraw.Draw(image)
                for face in faces:
                    box = face["bounding_box"]
                    xmin = box["x_min"]
                    ymin = box["y_min"]
                    xmax = xmin + box["width"]
                    ymax = ymin + box["height"]
                    
                    # Draw a red rectangle (width=3 pixels)
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                
                # Display the processed image
                st.image(image, caption="Detected Faces", use_container_width=True)
                
                # Show the raw JSON for verification
                with st.expander("View API Response"):
                    st.json(result)
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")