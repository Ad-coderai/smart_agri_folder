import cv2
import numpy as np
import json
import os
import tensorflow as tf

# Configuration
MODEL_PATH = 'plant_model.h5'
CLASSES_PATH = 'classes.json'

# Global variables to ensure the model is loaded only once
_MODEL = None
_CLASSES = None

def load_resources():
    """Lazy load: Loads model only when first needed."""
    global _MODEL, _CLASSES
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = tf.keras.models.load_model(MODEL_PATH)
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, 'r') as f:
                _CLASSES = json.load(f)
    return _MODEL, _CLASSES

def analyze_health(frame):
    """
    Analyzes a single frame provided by the Streamlit app.
    Returns: (label, confidence_score)
    """
    model, classes = load_resources()
    if model is None or classes is None:
        return "Model Not Found", 0

    try:
        # Preprocessing
        # Streamlit provides RGB, OpenCV is BGR. Adjust if colors look inverted.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        input_array = np.expand_dims(img.astype('float32') / 255.0, axis=0)

        # Inference
        predictions = model.predict(input_array, verbose=0)[0]
        max_idx = np.argmax(predictions)
        
        label = classes[max_idx]
        confidence = int(predictions[max_idx] * 100)
        
        return label, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0