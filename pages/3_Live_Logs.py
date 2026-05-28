import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Live System Logs", page_icon="📋", layout="wide")

st.title("📋 Live System Logs")
st.write("Real-time timestamped events from your Smart Agri-Assistant.")

LOG_FILE = "logs.txt"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp | Type | Event | Metric\n")

def load_logs():
    try:
        df = pd.read_csv(LOG_FILE, sep="|", names=["Timestamp", "Type", "Event", "Metric"], engine="python")
        return df.iloc[::-1] # Show latest first
    except Exception as e:
        return pd.DataFrame(columns=["Timestamp", "Type", "Event", "Metric"])

if st.button("Refresh Logs"):
    st.rerun()

df_logs = load_logs()

if not df_logs.empty:
    st.dataframe(df_logs, use_container_width=True)
else:
    st.info("No logs found yet. Start scanning to generate some activity!")

st.markdown("---")
if st.button("Clear Logs"):
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        st.success("Logs cleared!")
        st.rerun()
