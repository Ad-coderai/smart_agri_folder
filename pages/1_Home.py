import streamlit as st
import os

st.set_page_config(page_title="Smart Agri-Assistant - Home", page_icon="🌱", layout="wide")

st.title("🌱 Smart Agri-Assistant")
st.subheader("Welcome back, Farmer!")

col1, col2 = st.columns([1, 2])

with col1:
    # Mascot - using a placeholder if no image exists
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=200) 

with col2:
    st.markdown("""
    ### Today's Highlights
    - **System Status:** 🟢 Operational
    - **Sensors:** 3 active (Soil, Humidity, Temp)
    - **Latest Event:** Water pump triggered at 08:30 AM
    - **Crop Health:** 95% Healthy (Scanning in progress)
    """)
    
    st.info("💡 **Pro Tip:** Keep your sensors clean for more accurate readings.")

st.markdown("---")
st.write("Use the sidebar to navigate between Plant Health scanning and Live Logs.")
