import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
from PIL import Image
import numpy as np
from vision import analyze_health
from bridge import log_event, send_command

st.set_page_config(page_title="Plant Health Scanner", page_icon="🔍", layout="wide")

st.title("🔍 Plant Health & Disease Scanner")
st.write("Use the camera below to scan your plants for pests or disease.")

img_file_buffer = st.camera_input("Take a snapshot of the plant")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)
    
    # Analysis
    label, score = analyze_health(img_array)
    
    st.subheader(f"Results: {label}")
    st.progress(score / 100)
    st.write(f"Confidence/Severity Score: {score}%")
    
    if "Sick" in label or "Pest" in label:
        st.error("⚠️ PEST DETECTED!")
        if st.button("Trigger Emergency Spray"):
            if send_command("SPRAY"):
                st.success("Spray command sent to Arduino!")
                log_event("Action", "Manual Emergency Spray", "Success")
            else:
                st.error("Failed to connect to Arduino.")
    else:
        st.success("✅ Plant appears to be healthy!")
        log_event("Detection", "Manual scan: Healthy", "100%")

st.markdown("---")
st.subheader("Manual Overrides")
col1, col2 = st.columns(2)
with col1:
    if st.button("Manual Water"):
        send_command("WATER")
        st.info("Watering sequence started.")
with col2:
    if st.button("Manual Spray"):
        send_command("SPRAY")
        st.info("Spraying sequence started.")
