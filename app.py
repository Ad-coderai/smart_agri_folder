import streamlit as st
import pandas as pd
import os
import requests
import datetime
import time
import cv2
import numpy as np
from vision import analyze_health

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Agri-Assistant | Control Center", page_icon="🌱", layout="wide")

# API Configuration
WEATHER_API_KEY = "5b7d537e323710bd4efd365c0de04fe5"
MOISTURE_FILE = 'moisture_data.txt'
COMMAND_FILE = 'command.txt'
FERT_TASK_FILE = 'fert_task.txt'
LOG_FILE = 'logs.txt'

# Custom CSS
st.markdown("""
    <style>
    h1, h2, h3 { color: #1B5E20 !important; }
    .stApp { background-color: #F1F8E9; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED DATA HELPERS ---
def get_system_status():
    if os.path.exists(MOISTURE_FILE):
        last_mod = os.path.getmtime(MOISTURE_FILE)
        if (time.time() - last_mod) < 15:
            return "🟢 System Active", True
    return "🔴 Warning: Bridge Offline", False

def get_live_moisture():
    if os.path.exists(MOISTURE_FILE):
        try:
            with open(MOISTURE_FILE, 'r') as f:
                content = f.read().strip()
                return int(content) if content.isdigit() else 0
        except: pass
    return 0

def get_last_spray():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if any(x in line for x in ["SPRAY", "WATER", "FERTILIZER"]):
                        return line.split(']')[1].strip() if ']' in line else line
        except: pass
    return "No records"

# --- FRAGMENT: SIDEBAR LIVE STATS ---
@st.fragment(run_every=2)
def render_sidebar_stats():
    # Note: We do NOT use st.sidebar inside the fragment function.
    # We call this function WITHIN a 'with st.sidebar' block.
    status_text, is_active = get_system_status()
    moisture = get_live_moisture()
    st.markdown(f"### Status: {status_text}")
    st.metric(label="Live Moisture", value=f"{moisture}%", delta="Connected" if is_active else "Offline")

# --- SIDEBAR UI ---
st.sidebar.title("🌱 SproutSmart v2.7.1")

with st.sidebar:
    render_sidebar_stats()

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Field Climate")
city = st.sidebar.text_input("City", "Bengaluru")
if st.sidebar.button("Get Weather"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()
        if data["cod"] == 200:
            st.sidebar.success(f"{data['main']['temp']}°C | {data['weather'][0]['description'].title()}")
            st.sidebar.write(f"Humidity: {data['main']['humidity']}%")
        else:
            st.sidebar.error("City not found")
    except:
        st.sidebar.error("Connection Error")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Dashboard", "🎥 Live AI Vision", "📋 Audit Logs"])
st.sidebar.caption("v2.7.1-stable | Fragments Patched")

# --- CAMERA INITIALIZATION ---
if 'vc' not in st.session_state:
    st.session_state.vc = cv2.VideoCapture(1)

# --- FRAGMENT: LIVE VIDEO FEED ---
@st.fragment(run_every=0.1)
def live_vision_feed():
    frame_placeholder = st.empty()
    health_status = st.empty()
    
    if st.session_state.vc.isOpened():
        ret, frame = st.session_state.vc.read()
        if ret:
            label, score = analyze_health(frame)
            color = (0, 255, 0) if "Healthy" in label else (0, 0, 255)
            cv2.putText(frame, f"{label} ({score}%)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)
            
            if "Healthy" in label:
                health_status.success(f"AI Detection: {label} ({score}%) - All clear.")
            else:
                health_status.error(f"AI Alert: {label} detected! Confidence: {score}%")
                if score > 85:
                    with open(COMMAND_FILE, 'w') as f: f.write('SPRAY')
    else:
        st.error("Camera error. Reconnecting...")
        st.session_state.vc = cv2.VideoCapture(0)

# --- 1. DASHBOARD PAGE ---
if page == "🏠 Dashboard":
    st.title("🌱 Smart Garden: Soil sweetness")
    st.markdown("##### Soil moisture is based on your local Arduino sensor.")
    
    @st.fragment(run_every=2)
    def dashboard_main():
        moisture = get_live_moisture()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Moisture Level", value=f"{moisture}%")
        with col2:
            st.metric(label="Health Score", value="94%" if moisture > 30 else "62%")
        with col3:
            st.metric(label="Recent Action", value=get_last_spray()[:25] + "...")

        st.markdown("---")
        
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.markdown("### Plant Mascot")
            if moisture >= 30:
                st.image('https://cdn-icons-png.flaticon.com/512/2917/2917995.png', width=180)
                st.success("✅ Soil moist and happy.")
            else:
                st.image('https://cdn-icons-png.flaticon.com/512/628/628283.png', width=180)
                st.warning("🚨 Needs sweetness.")

        with c_right:
            st.markdown("### 🕹️ Hardware Overrides")
            if st.button("🚀 TRIGGER SPRAY", use_container_width=True):
                with open(COMMAND_FILE, 'w') as f: f.write('SPRAY')
                st.toast("Spray command sent!")
            if st.button("💧 TRIGGER WATER", use_container_width=True):
                with open(COMMAND_FILE, 'w') as f: f.write('WATER')
                st.toast("Water command sent!")
            if st.button("🧪 TRIGGER FERTILIZER", use_container_width=True):
                with open(FERT_TASK_FILE, 'w') as f: f.write('F')
                st.toast("Fertilizer command sent!")
                
    dashboard_main()

# --- 2. LIVE AI VISION ---
elif page == "🎥 Live AI Vision":
    st.title("🎥 Real-Time AI Diagnostics")
    live_vision_feed()

# --- 3. AUDIT LOGS ---
elif page == "📋 Audit Logs":
    st.title("📋 System Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            st.text_area("Audit Log", value=f.read(), height=500)
        if st.button("Clear Logs"):
            os.remove(LOG_FILE)
            st.rerun()
