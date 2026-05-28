import streamlit as st
import os
import requests
import time
import cv2
from vision import analyze_health # Ensure this file exists

st.set_page_config(page_title="Smart Agri-Assistant", layout="wide")

# API Configuration
WEATHER_API_KEY = "5b7d537e323710bd4efd365c0de04fe5"

# --- HELPERS ---
def get_live_moisture():
    if os.path.exists('moisture_data.txt'):
        try:
            with open('moisture_data.txt', 'r') as f:
                return f.read().strip()
        except: return "0"
    return "Offline"

@st.cache_data(ttl=600)
def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        return requests.get(url).json()
    except: return None

# --- UI ---
st.title("🌱 SproutSmart Dashboard")

# Sidebar
with st.sidebar:
    st.subheader("Field Climate")
    city = st.text_input("City", "Bengaluru")
    if st.button("Update Weather"):
        w = get_weather(city)
        if w: st.success(f"{w['main']['temp']}°C | {w['weather'][0]['description']}")
        else: st.error("Weather unavailable")

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    @st.fragment(run_every=2)
    def show_moisture():
        st.metric("Soil Moisture", f"{get_live_moisture()}%")
    show_moisture()

    if st.button("💧 Trigger Water"):
        with open('command.txt', 'w') as f: f.write('WATER')

with col2:
    st.subheader("AI Vision")
    # Camera Logic
    if 'vc' not in st.session_state:
        st.session_state.vc = cv2.VideoCapture(1)
    
    @st.fragment(run_every=1) # Slower refresh for AI to prevent CPU lag
    def run_vision():
        ret, frame = st.session_state.vc.read()
        if ret:
            label, score = analyze_health(frame)
            st.image(frame, channels="BGR", use_container_width=True)
            st.write(f"Detection: {label} ({score}%)")
    run_vision()