import streamlit as st
from ultralytics import YOLO
import gdown
import os

# Google Drive file ID from your link
FILE_ID = "1qM0-5Ca55hyuGTtuAafxqtlbw0Z97-5p"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    # If the model file doesn't exist locally, download it from Google Drive
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    return YOLO(MODEL_PATH)

# Load the model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Rest of your Streamlit application UI code goes here...
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🛡️ Real-Time Construction Safety Detection</h1>", unsafe_allow_html=True)
st.write("Upload an image or video to detect safety gear compliance.")
