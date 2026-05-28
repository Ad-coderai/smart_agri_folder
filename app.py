import streamlit as st
import pandas as pd
import os
import requests
import datetime
import time
import cv2
import numpy as np
import random
from vision import analyze_health

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sprout Smart | Control Center", page_icon="🌱", layout="wide")

# Constants
MOISTURE_FILE = 'moisture_data.txt'
COMMAND_FILE = 'command.txt'
FERT_TASK_FILE = 'fert_task.txt'
LOG_FILE = 'logs.txt'

# Custom CSS for Aesthetic Theme
st.markdown("""
    <style>
    h1, h2, h3 { color: #1B5E20 !important; font-family: 'Segoe UI', sans-serif; }
    .stApp { background-color: #F1F8E9; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .instruction-card { background: #E8F5E9; padding: 20px; border-radius: 15px; border: 1px solid #C8E6C9; margin-bottom: 10px; }
    .plant-card { background: white; padding: 10px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 5px; }
    .stToggle { background: #E8F5E9; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA HELPERS ---
def get_system_status():
    if os.path.exists(MOISTURE_FILE):
        last_mod = os.path.getmtime(MOISTURE_FILE)
        return ("🟢 System Active", True) if (time.time() - last_mod) < 20 else ("🔴 Warning: Bridge Offline", False)
    return "⚪ Not Connected", False

def get_live_moisture():
    if os.path.exists(MOISTURE_FILE):
        try:
            with open(MOISTURE_FILE, 'r') as f:
                content = f.read().strip()
                return int(content) if content.isdigit() else 0
        except: pass
    return 0

def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, sep="|", names=["Timestamp", "Type", "Event", "Metric"], engine="python")
            return df.iloc[::-1]
        except: return pd.DataFrame(columns=["Timestamp", "Type", "Event", "Metric"])
    return pd.DataFrame(columns=["Timestamp", "Type", "Event", "Metric"])

# --- SESSION STATE INITIALIZATION ---
if 'plants' not in st.session_state:
    st.session_state.plants = []
if 'vc' not in st.session_state:
    st.session_state.vc = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = 0

# --- SHARED STATE ---
status_text, is_active = get_system_status()
moisture = get_live_moisture()

# --- SIDEBAR: PERSONALIZATION & NAVIGATION ---
with st.sidebar:
    st.title("🌱 SproutSmart v4.2")
    
    if 'farm_name' not in st.session_state:
        st.session_state.farm_name = "My Smart Farm"
    st.session_state.farm_name = st.text_input("Name your Farm/Garden:", st.session_state.farm_name)
    st.markdown(f"### 📍 {st.session_state.farm_name}")
    
    @st.fragment(run_every=2)
    def sidebar_live():
        f_status, _ = get_system_status()
        f_moisture = get_live_moisture()
        st.markdown(f"**Status:** {f_status}")
        st.metric(label="Live Moisture", value=f"{f_moisture}%")
    sidebar_live()
    
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🌿 My Plants", "🎥 Auto-Scan & Control", "📋 Activity Log"])
    st.markdown("---")
    st.caption("Powered by SproutSmart AI & IoT")

# --- 1. HOME PAGE ---
if page == "🏠 Home":
    st.title(f"Welcome to {st.session_state.farm_name}")
    
    if 'current_quote' not in st.session_state:
        quotes = [
            "\"The glory of gardening: hands in the dirt, head in the sun, heart with nature.\"",
            "\"To plant a garden is to believe in tomorrow.\"",
            "\"The best time to plant a tree was 20 years ago. The second best time is now.\"",
            "\"A garden requires patient labor and attention.\""
        ]
        st.session_state.current_quote = random.choice(quotes)
    st.markdown(f"*{st.session_state.current_quote}*")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if moisture >= 30:
            st.image('https://cdn-icons-png.flaticon.com/512/2917/2917995.png', width=250)
            st.success("✅ **Status:** Soil moist and happy. Keep the roots cozy.")
        else:
            st.image('https://cdn-icons-png.flaticon.com/512/628/628283.png', width=250)
            st.warning("🚨 **Alert:** Soil is dry! Needs some sweetness.")
            
    with col2:
        st.markdown("### Farm Overview")
        m1, m2 = st.columns(2)
        m1.metric("Moisture", f"{moisture}%", delta="Stable" if moisture > 30 else "Critical")
        m2.metric("Plants Managed", len(st.session_state.plants))
        
        st.info("💡 **Farmer's Tip:** Healthy plants start with healthy soil and consistent monitoring.")

# --- 2. MY PLANTS (Instructions & Setup) ---
elif page == "🌿 My Plants":
    st.title("🌿 Plant Management")
    st.write("Register your plants and follow the setup guide.")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("➕ Add New Plant")
        with st.form("add_plant_form", clear_on_submit=True):
            new_p_name = st.text_input("Plant Name:", placeholder="e.g. Cherry Tomato")
            new_p_type = st.selectbox("Type:", ["Vegetable", "Flower", "Fruit", "Herb", "Other"])
            submitted = st.form_submit_button("Enter / Save Plant")
            if submitted and new_p_name:
                st.session_state.plants.append({"name": new_p_name, "type": new_p_type, "date": datetime.date.today().strftime("%Y-%m-%d")})
                st.success(f"Added {new_p_name} to your garden!")
        
        st.markdown("#### 📋 Setup Instructions:")
        st.markdown("""
        <div class='instruction-card'>
        1. 📍 <b>Place Device:</b> Insert your moisture sensor deep into the root zone.<br>
        2. 💧 <b>Prepare Bucket:</b> Add fertilizer or water to the connected reservoir.<br>
        3. ⚙️ <b>Auto-Sync:</b> Once enabled, the AI scans and the hardware reacts based on the interval.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("🏡 Registered Plants")
        if st.session_state.plants:
            for i, p in enumerate(st.session_state.plants):
                st.markdown(f"""
                <div class='plant-card'>
                <b>{p['name']}</b> ({p['type']})<br>
                <small>Added on: {p['date']}</small>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Clear All Plants"):
                st.session_state.plants = []
                st.rerun()
        else:
            st.info("No plants registered yet. Use the form on the left to add one!")

# --- 3. AUTO-SCAN & CONTROL ---
elif page == "🎥 Auto-Scan & Control":
    st.title("🎥 Control Center")
    
    if st.session_state.vc is None:
        st.session_state.vc = cv2.VideoCapture(0) # Default to 0 for broader compatibility
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Live Viewport")
        auto_scan = st.toggle("🤖 Enable Automatic AI Scanning", value=False)
        scan_interval = st.slider("Scan Frequency (Seconds):", 1, 60, 5)
        
        @st.fragment(run_every=0.1) # UI refresh every 100ms for smooth video
        def live_feed_fragment():
            if st.session_state.vc and st.session_state.vc.isOpened():
                ret, frame = st.session_state.vc.read()
                if ret:
                    current_time = time.time()
                    
                    # 1. AI Detection Logic (Only runs on interval)
                    if auto_scan and (current_time - st.session_state.last_scan_time) >= scan_interval:
                        label, score = analyze_health(frame)
                        st.session_state.last_scan_time = current_time
                        
                        # Trigger hardware based on AI/Sensor
                        if "Healthy" not in label and score > 80:
                            with open(COMMAND_FILE, 'w') as f: f.write('SPRAY')
                            st.toast(f"AI Detected {label}: Spraying!", icon="🚿")
                        elif moisture < 30:
                            with open(COMMAND_FILE, 'w') as f: f.write('WATER')
                            st.toast("Low Moisture: Watering!", icon="💧")
                    
                    # 2. Display live feed always
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, use_container_width=True)
                    
                    # 3. Status text overlay
                    if auto_scan:
                        st.caption(f"🤖 AI Scanning Active | Interval: {scan_interval}s")
                else:
                    st.error("Failed to capture frame. Retrying...")
            else:
                st.error("Camera not initialized. Try restarting bridge or app.")
        
        live_feed_fragment()

    with c2:
        st.subheader("Manual Overrides")
        if st.button("🚀 TRIGGER SPRAY (Pests/Sick)", use_container_width=True):
            with open(COMMAND_FILE, 'w') as f: f.write('SPRAY')
            st.toast("Manual Spray Sent!")
        if st.button("💧 TRIGGER WATER (Dry)", use_container_width=True):
            with open(COMMAND_FILE, 'w') as f: f.write('WATER')
            st.toast("Manual Water Sent!")
        if st.button("🧪 TRIGGER FERTILIZER", use_container_width=True):
            with open(FERT_TASK_FILE, 'w') as f: f.write('F')
            st.toast("Fertilizer command sent!")
        
        st.markdown("---")
        st.subheader("🔍 One-Time Scan")
        if st.button("📸 Capture & Detect Now"):
            ret, frame = st.session_state.vc.read()
            if ret:
                label, score = analyze_health(frame)
                st.success(f"Result: **{label}** ({score}%)")
                if "Healthy" not in label:
                    st.warning("Action suggested: Spray fertilizer.")

# --- 4. ACTIVITY LOG ---
elif page == "📋 Activity Log":
    st.title("📋 Live System Logs")
    
    @st.fragment(run_every=2)
    def show_logs():
        df = load_logs()
        if not df.empty:
            st.table(df.head(25))
            if st.button("🗑️ Clear Audit Trail"):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                    st.rerun()
        else:
            st.info("No activity logged yet. Start scanning to see events here.")
    show_logs()
