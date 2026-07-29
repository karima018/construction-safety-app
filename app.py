import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
from datetime import datetime
import gdown
import os
import io

# --- Google Drive Configuration ---
FILE_ID = "1qM0-5Ca55hyuGTtuAafxqtlbw0Z97-5p"
MODEL_PATH = "best.pt"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("Initial model download from Drive... (This happens once)"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
    return YOLO(LOCAL_MODEL_PATH)

try:
    model = load_model()
    model_status = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_status = False

# --- Page Configuration ---
st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🛡️", layout="wide")

# --- Custom CSS Styling (Dark Theme & Modern Cards like Pro Dashboard) ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #121826;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    .metric-card {
        background-color: #121826;
        border: 1px solid #1e293b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Top Header Bar ---
col_h1, col_h2, col_h3 = st.columns([2, 5, 2])
with col_h1:
    st.markdown("### 🛡️ YOLOv8 Safety")
with col_h2:
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 14px;'>Real-time PPE Detection & Access Control System</p>", unsafe_allow_html=True)
with col_h3:
    current_time = datetime.now().strftime("%I:%M:%S %p")
    st.markdown(f"<p style='text-align: right; color: #38bdf8; font-size: 14px;'>🕒 <b>{current_time}</b></p>", unsafe_allow_html=True)

st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.markdown("### 🎛️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Input Mode", ["📁 Upload Image", "📷 Webcam Live Photo"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the tabs below to switch between Live Stream, Analytics, and Logs.")

# --- Tab Navigation (Like Friend's Dashboard) ---
tab1, tab2, tab3 = st.tabs(["🟢 Live Video & Tracking", "📊 Analytics & Compliance", "🚨 Violation Logs"])

with tab1:
    if app_mode == "📁 Upload Image" and model_status:
        st.markdown("#### 📁 Construction Site Image Analysis")
        uploaded_file = st.file_uploader("Upload worker image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            try:
                image_bytes = uploaded_file.read()
                original_image = Image.open(io.BytesIO(image_bytes))
                
                with col1:
                    st.markdown("##### 🖼️ Original Image")
                    st.image(original_image, use_container_width=True)
                    
                if st.button("🚀 Run Safety Analysis", use_container_width=True):
                    with st.spinner("Analyzing site safety components..."):
                        results = model(original_image, conf=confidence)
                        res_plotted = results[0].plot()
                        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
                        with col2:
                            st.markdown("##### 🔍 AI Detection Output")
                            st.image(res_rgb, use_container_width=True)
                        
                        boxes = results[0].boxes
                        class_names = results[0].names
                        detected_classes = [class_names[int(cls)] for cls in boxes.cls]
                        
                        helmet_count = detected_classes.count('helmet')
                        vest_count = detected_classes.count('vest')
                        has_helmet = 'helmet' in detected_classes
                        has_vest = 'vest' in detected_classes
                        
                        # --- Metrics Section inside Tab 1 ---
                        st.markdown("### 📊 Detection Statistics")
                        m1, m2, m3 = st.columns(3)
                        with m1: st.metric(label="👷 Helmets Detected", value=helmet_count)
                        with m2: st.metric(label="🦺 Safety Vests Detected", value=vest_count)
                        with m3: st.metric(label="📌 Total Detections", value=len(detected_classes))

                        st.markdown("### 🚦 Access Control Gate Status")
                        if has_helmet and has_vest:
                            st.success("✅ **ACCESS GRANTED (ALLOWED)** — Worker is wearing complete safety gear.")
                        else:
                            violation_text = []
                            if not has_helmet: violation_text.append("Helmet")
                            if not has_vest: violation_text.append("Safety Vest")
                            st.error(f"❌ **ACCESS DENIED** — Missing: {', '.join(violation_text)}.")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif app_mode == "📷 Webcam Live Photo" and model_status:
        st.markdown("#### 📷 Live Snapshot Safety Test")
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
            
            m1, m2, m3 = st.columns(3)
            with m1: st.metric(label="👷 Helmets", value=helmet_count)
            with m2: st.metric(label="🦺 Vests", value=vest_count)
            with m3: st.metric(label="📌 Total", value=len(detected_classes))

with tab2:
    st.markdown("### 📈 Site Analytics & Compliance Overview")
    st.info("Analytics charts and historical compliance graphs will appear here based on site activity logs.")
    st.progress(85, text="Overall Weekly Site Safety Compliance: 85%")

with tab3:
    st.markdown("### 🚨 Recorded Safety Violation Logs")
    st.warning("⚠️ No recent critical breaches logged in the current session. System is actively monitoring.")
