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

# Directory for model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_PATH)

@st.cache_resource
def load_model():
    # Download model from Google Drive if not exists
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("Initial model download from Drive... (This happens once)"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
    
    return YOLO(LOCAL_MODEL_PATH)

# Load model
try:
    model = load_model()
    model_status = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_status = False

# --- Page Configuration ---
st.set_page_config(page_title="Construction Safety Dashboard", page_icon="🛡️", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2130; padding: 15px;
        border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🛡️ Real-Time Construction Safety & Access Control</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Automated PPE Detection & Site Security Dashboard</p>", unsafe_allow_html=True)

current_time = datetime.now().strftime("%B %d, %Y | %I:%M:%S %p")
st.markdown(f"<p style='text-align: center; color: #00ffcc; font-size: 14px;'>📅 Current System Date & Time: <b>{current_time}</b></p>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Design ---
st.sidebar.markdown("### 🎛️ Control Panel")
app_mode = st.sidebar.selectbox("Choose Input Mode", ["📁 Upload Image", "📷 Webcam Live Photo"])
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust the confidence slider if items are not detected properly.")

if app_mode == "📁 Upload Image" and model_status:
    st.subheader("📁 Construction Site Image Analysis")
    uploaded_file = st.file_uploader("Upload worker image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        try:
            image_bytes = uploaded_file.read()
            original_image = Image.open(io.BytesIO(image_bytes))
            
            # Attempt to fix orientation based on EXIF tags
            from PIL import ExifTags
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                exif = dict(original_image._getexif().items())

                if exif[orientation] == 3: original_image = original_image.rotate(180, expand=True)
                elif exif[orientation] == 6: original_image = original_image.rotate(270, expand=True)
                elif exif[orientation] == 8: original_image = original_image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            with col1:
                st.markdown("#### 🖼️ Original Image")
                st.image(original_image, use_container_width=True)
                
            if st.button("🚀 Run Safety Analysis", use_container_width=True):
                with st.spinner("Analyzing site safety components..."):
                    results = model(original_image, conf=confidence)
                    res_plotted = results[0].plot()
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.markdown("#### 🔍 AI Detection Output")
                        st.image(res_rgb, use_container_width=True)
                    
                    boxes = results[0].boxes
                    class_names = results[0].names
                    detected_classes = [class_names[int(cls)] for cls in boxes.cls]
                    
                    helmet_count = detected_classes.count('helmet')
                    vest_count = detected_classes.count('vest')
                    
                    st.markdown("### 📊 Detection Statistics")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1: st.metric(label="👷 Helmets Detected", value=helmet_count)
                    with col_m2: st.metric(label="🦺 Safety Vests Detected", value=vest_count)
                    with col_m3: st.metric(label="📌 Total Detections", value=len(detected_classes))

                    has_helmet = 'helmet' in detected_classes
                    has_vest = 'vest' in detected_classes
                    
                    safety_score = 100 if (has_helmet and has_vest) else 50
                    st.markdown("### 📈 Compliance Meter")
                    st.progress(safety_score, text=f"Safety Compliance Score: {safety_score}%")

                    st.markdown("### 🚦 Access Control Gate Status")
                    if has_helmet and has_vest:
                        st.success("✅ **ACCESS GRANTED (ALLOWED)** — Worker is wearing complete safety gear and complies with site policies.")
                    else:
                        violation_text = []
                        if not has_helmet: violation_text.append("Helmet")
                        if not has_vest: violation_text.append("Safety Vest")
                        
                        st.error(f"❌ **ACCESS DENIED (NOT ALLOWED)** — Safety Violation Detected! Missing: {', '.join(violation_text)}.")
                        st.warning("⚠️ **ALERT:** Immediate disciplinary action or supervisor check required for this worker.")

        except Exception as e:
            st.error(f"Error processing image: {e}")

elif app_mode == "📷 Webcam Live Photo" and model_status:
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
        with col_m1: st.metric(label="👷 Helmets Detected", value=helmet_count)
        with col_m2: st.metric(label="🦺 Safety Vests Detected", value=vest_count)
        with col_m3: st.metric(label="📌 Total Detections", value=len(detected_classes))
            
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

elif not model_status:
    st.warning("Waiting for model to load...")
