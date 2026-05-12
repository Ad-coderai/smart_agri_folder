import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# --- 1. SET UP THE PAGE FIRST ---
st.set_page_config(page_title="Smart Agri-Tech", page_icon="🌱")
st.title("🌱 Smart Agri-Tech Disease Detector")

# --- 2. THE "LAZY" LOADERS ---
@st.cache_resource
def load_ai_model():
    # Only runs once and stays in memory
    if os.path.exists('plant_model.h5'):
        return tf.keras.models.load_model('plant_model.h5')
    return None

@st.cache_resource
def load_labels():
    if os.path.exists('classes.json'):
        with open("classes.json", "r") as f:
            return json.load(f)
    return None

# --- 3. UI LAYOUT ---
st.info("System Ready. Please upload an image to begin analysis.")

img_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Target Leaf", use_container_width=True)
    
    # Only load the model WHEN an image is provided
    with st.spinner('Loading AI Model...'):
        model = load_ai_model()
        labels = load_labels()

    if model is None or labels is None:
        st.error("Error: 'plant_model.h5' or 'classes.json' missing from folder!")
    else:
        # --- 4. PREDICTION LOGIC ---
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        detected_class = labels[result_index]
        
        st.subheader(f"Result: {detected_class}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        if "healthy" in detected_class.lower():
            st.success("Plant is Healthy. No action needed.")
        else:
            st.warning(f"Disease Detected! (Hardware trigger signal: S1_P)")
            st.info("Note: Connect Arduino to enable physical spraying.")