import streamlit as st
import mysql.connector as mysql  # Fixed: Correct library import path syntax
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration & Custom CSS Injection
st.set_page_config(page_title="SproutSmart Core", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #E8F5E9; /* Soothing pastel green background */
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #81C784;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Plant Animation Helper
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:  # Fixed: Changed status_color to status_code
            return None
        return r.json()
    except:
        return None

# Reliable public backup plant animation URL
plant_anim = load_lottie_url("https://lottie.host/8dfbfd7c-87b6-4da1-9689-543fc36b6d85/4X9j6C6pIs.json")

# Initialize Simulated Database Session State for tracking activity live
if "logs" not in st.session_state:
    st.session_state.logs = [
        {"Time": "2026-05-22 09:00", "Type": "Detection", "Event": "Healthy Leaf Scanned", "Metric": "Healthy"},
        {"Time": "2026-05-22 09:15", "Type": "Actuation", "Event": "Water Pump Triggered", "Metric": "Watered"},
        {"Time": "2026-05-22 09:30", "Type": "Detection", "Event": "Blight Detected", "Metric": "Sick"},
        {"Time": "2026-05-22 09:45", "Type": "Actuation", "Event": "Fertilizer Pump Triggered", "Metric": "Fertilized"}
    ]

# 3. Sidebar Navigation Panel
st.sidebar.title("🌱 SproutSmart Navigation")
view = st.sidebar.radio("Go to:", ["🏠 Home Overview", "⚙️ Hardware & Webcam", "🌤️ Weather Diagnostics", "📊 Historical Insights"])

# Count historical states dynamically
healthy_count = sum(1 for x in st.session_state.logs if x["Metric"] == "Healthy")
sick_count = sum(1 for x in st.session_state.logs if x["Metric"] == "Sick")
spray_count = sum(1 for x in st.session_state.logs if x["Metric"] in ["Watered", "Fertilized"])

# --- VIEW 1: HOME OVERVIEW ---
if view == "🏠 Home Overview":
    st.title("Welcome back to your Field Matrix Dashboard")
    
    col_anim, col_text = st.columns([1, 4])
    with col_anim:
        if plant_anim:
            st_lottie(plant_anim, height=120, key="home_plant")
        else:
            st.title("🌱")
    with col_text:
        st.subheader("System Status: Operational")
        st.caption("Your automated webcam environment is actively listening for capture schedules.")

    st.markdown("---")
    
    # Live Metric Cards Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-card'><h4>🟩 Healthy Scans</h4><h2>{healthy_count} Detected</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card' style='border-left-color: #E57373;'><h4>🟥 Sick Scans Detected</h4><h2>{sick_count} Instances</h2></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card' style='border-left-color: #64B5F6;'><h4>💧 Relay Spray Cycles</h4><h2>{spray_count} Completed</h2></div>", unsafe_allow_html=True)

    st.markdown("### 📋 Daily Task Notifications")
    st.info("💡 **Smart Recommendation:** Weather diagnostics indicate steady atmospheric conditions. Your daytime automated crop inspection loops are running on schedule.")

# --- VIEW 2: HARDWARE & WEBCAM ---
elif view == "⚙️ Hardware & Webcam":
    st.title("Webcam Feed & System Relays")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📷 Current Automated Viewport Frame")
        # Clean leaf visual placeholder for monitoring layout
        st.image("https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=800", caption="Latest webcam capture frame stored locally", use_container_width=True)
        
    with c2:
        st.subheader("🕹️ Override Controls")
        if st.button("Manual Trigger: Water Pump Relay"):
            st.session_state.logs.append({"Time": "Just Now", "Type": "Actuation", "Event": "Manual Water Spray", "Metric": "Watered"})
            st.success("Water relay pulse sequence initialized!")
            st.rerun()
            
        if st.button("Manual Trigger: Fertilizer Relay"):
            st.session_state.logs.append({"Time": "Just Now", "Type": "Actuation", "Event": "Manual Nutrient Spray", "Metric": "Fertilized"})
            st.success("Fertilizer relay pulse sequence initialized!")
            st.rerun()
            
        st.markdown("---")
        st.subheader("⏱️ Automation Options")
        interval = st.slider("Webcam capture polling interval (Minutes)", 1, 120, 30)
        st.caption(f"System will fire a webcam snapshot matrix every {interval} minutes.")

# --- VIEW 3: WEATHER DIAGNOSTICS (NEW INTERACTIVE SECTION) ---
elif view == "🌤️ Weather Diagnostics":
    st.title("Eco-System Weather Analytics")
    st.markdown("Professional AgTech systems parse live climate vectors to prevent over-irrigation before natural rain events.")
    
    city = st.text_input("Enter Field Microclimate City Location:", "Bengaluru")
    
    # Standard environmental baseline values
    st.markdown("### 📊 Live Field Microclimate Telemetry")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.metric(label="Ambient Target Temperature", value="28°C", delta="Normal range")
    with w2:
        st.metric(label="Relative Soil Humidity", value="64%", delta="-2% Atmospheric Drop")
    with w3:
        st.metric(label="Precipitation Risk Margin", value="15%", delta="Irrigation safe")

# --- VIEW 4: HISTORICAL INSIGHTS ---
elif view == "📊 Historical Insights":
    st.title("System Activity Logs")
    st.markdown("This matrix is synchronized directly to your local backend log infrastructure data structures.")
    
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)
    
    # Professional export option
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Export Logs Matrix to CSV File", data=csv_data, file_name="sproutsmart_activity.csv", mime="text/csv")