import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
from datetime import datetime
import gdown
import os

# Google Drive file ID from your link
FILE_ID = "1qM0-5Ca55hyuGTtuAafxqtlbw0Z97-5p"
MODEL_PATH = "best.pt"

# Page Configuration
st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS Styling for Modern UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

# Load model safely
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Header Section with Styling
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🛡️ Real-Time Construction Safety & Access Control</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Automated PPE Detection & Site Security Dashboard</p>", unsafe_allow_html=True)

# --- Added Feature: Live Date & Time Display ---
current_time = datetime.now().strftime("%B %d, %Y | %I:%M:%S %p")
st.markdown(f"<p style='text-align: center; color: #00ffcc; font-size: 14px;'>📅 Current System Date & Time: <b>{current_time}</b></p>", unsafe_allow_html=True)
# -----------------------------------------------

st.markdown("---")

# Sidebar Design
st.sidebar.markdown("### 🎛️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Input Mode", ["📁 Upload Image", "📷 Webcam Live Photo"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust the confidence slider if items are not detected properly.")

if app_mode == "📁 Upload Image":
    st.subheader("📁 Construction Site Image Analysis")
    uploaded_file = st.file_uploader("Upload worker image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(image, use_container_width=True)
            
        if st.button("🚀 Run Safety Analysis", use_container_width=True):
            with st.spinner("Analyzing site safety components..."):
                results = model(image, conf=confidence)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.markdown("#### 🔍 AI Detection Output")
                    st.image(res_rgb, use_container_width=True)
                
                boxes = results[0].boxes
                class_names = results[0].names
                detected_classes = [class_names[int(cls)] for cls in boxes.cls]
                
                # --- Metrics Display ---
                helmet_count = detected_classes.count('helmet')
                vest_count = detected_classes.count('vest')
                
                st.markdown("### 📊 Detection Statistics")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(label="👷 Helmets Detected", value=helmet_count)
                with col_m2:
                    st.metric(label="🦺 Safety Vests Detected", value=vest_count)
                with col_m3:
                    st.metric(label="📌 Total Detections", value=len(detected_classes))

                has_helmet = 'helmet' in detected_classes
                has_vest = 'vest' in detected_classes
                
                # --- Safety Progress Bar ---
                safety_score = 100 if (has_helmet and has_vest) else 50
                st.markdown("### 📈 Compliance Meter")
                st.progress(safety_score, text=f"Safety Compliance Score: {safety_score}%")

                # --- Access Control Status ---
                st.markdown("### 🚦 Access Control Gate Status")
                if has_helmet and has_vest:
                    st.success("✅ **ACCESS GRANTED (ALLOWED)** — Worker is wearing complete safety gear and complies with site policies.")
                else:
                    st.error("❌ **ACCESS DENIED (NOT ALLOWED)** — Safety Violation Detected! Missing Helmet and/or Vest.")
                    st.warning("⚠️ **ALERT:** Immediate disciplinary action or supervisor check required for this worker.")

elif app_mode == "📷 Webcam Live Photo":
    st.subheader("📷 Live Snapshot Safety Test")
    camera_file = st.camera_input("Take a photo of the worker")
    
    if camera_file is not None:
        image = Image.open(camera_file)
        results = model(image, conf=confidence)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        st.image(res_rgb, caption="Live Processed Frame", use_container_width=True)
        
        boxes = results[0].boxes
        class_names = results[0].names
        detected_classes = [class_names[int(cls)] for cls in boxes.cls]
        
        helmet_count = detected_classes.count('helmet')
        vest_count = detected_classes.count('vest')
        
        st.markdown("### 📊 Detection Statistics")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="👷 Helmets Detected", value=helmet_count)
        with col_m2:
            st.metric(label="🦺 Safety Vests Detected", value=vest_count)
        with col_m3:
            st.metric(label="📌 Total Detections", value=len(detected_classes))
            
        has_helmet = 'helmet' in detected_classes
        has_vest = 'vest' in detected_classes
        
        safety_score = 100 if (has_helmet and has_vest) else 50
        st.markdown("### 📈 Compliance Meter")
        st.progress(safety_score, text=f"Safety Compliance Score: {safety_score}%")
        
        st.markdown("### 🚦 Access Control Gate Status")
        if has_helmet and has_vest:
            st.success("✅ **STATUS: ALLOWED TO ENTER** — Complete gear verified.")
        else:
            st.error("❌ **STATUS: ACCESS DENIED** — Missing safety equipment!")
            st.warning("⚠️ **ALERT:** Immediate disciplinary action or supervisor check required for this worker.")
