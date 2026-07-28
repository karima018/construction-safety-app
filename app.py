import streamlit as st
from ultralytics import YOLO
import gdown
import os
from PIL import Image
import numpy as np

# Google Drive file ID from your link
FILE_ID = "1qM0-5Ca55hyuGTtuAafxqtlbw0Z97-5p"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

# Load model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# UI Header
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🛡️ Real-Time Construction Safety Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image to detect safety gear compliance.")

# File uploader option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Detect Safety Gear"):
        with st.spinner("Detecting..."):
            # Run YOLO prediction
            results = model(image)
            res_plotted = results[0].plot()
            
            # Display result
            st.image(res_plotted, caption='Processed Image with Detections', use_container_width=True)
            st.success("Detection Completed!")

