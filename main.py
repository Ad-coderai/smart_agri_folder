import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
import time
import datetime
import requests
import json
import io
import base64
import threading
import os
from PIL import Image
from typing import List

# Live moisture dashboard bridge (Arduino → JSON via data_collector.py)
try:
    from data_collector import get_latest_moisture_value
except Exception:  # keep app usable even if collector isn't started / import fails
    get_latest_moisture_value = None



# ─────────────────────────────────────────────
# APPLICATION STRUCTURE
# ─────────────────────────────────────────────
# 1. Sensor + camera helpers
# 2. Database helpers
# 3. Session state bootstrap
# 4. UI components and view renderers
# 5. Main application router


def read_arduino_serial_value(port: str = "COM3", baudrate: int = 9600, timeout: float = 1.0):
    """Read one line from the Arduino serial port and return it as an integer.

    Returns None if the port is unavailable, the data is empty, or the value
    is not an integer.
    """
    try:
        import serial
    except ImportError:
        return None

    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as ser:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                return None
            return int(raw_line)
    except (serial.SerialException, ValueError, OSError):
        return None


def capture_frame(camera_index: int = 0):
    """Capture one frame from the webcam and return an RGB image."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, "no_device"
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None, "capture_fail"

    b, g, r = cv2.split(frame)
    g_boosted = cv2.add(g, 15)
    frame_enhanced = cv2.merge([b, g_boosted, r])

    ts_text = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(
        frame_enhanced, f"SproutSmart Vision  |  {ts_text}",
        (10, frame_enhanced.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (129, 199, 132), 1, cv2.LINE_AA
    )

    h, w = frame_enhanced.shape[:2]
    cv2.rectangle(frame_enhanced, (w//4, h//4), (3*w//4, 3*h//4), (129, 199, 132), 2)
    cv2.putText(frame_enhanced, "SCAN ZONE", (w//4 + 5, h//4 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (129, 199, 132), 1)

    rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
    return rgb, "ok"


def perform_scan(camera_index: int = 0, moisture_value: int = None, auto: bool = False):
    """Capture a frame, analyze plant health, and optionally trigger automation."""
    frame, status = capture_frame(camera_index)
    if frame is None:
        return None, status, None, None

    classification, score = analyze_plant_health(frame)
    if classification == "Healthy":
        insert_log("Detection", "Live plant scan completed — plant identified as healthy", "Healthy")
    elif classification.startswith("Sick"):
        insert_log("Detection", f"Live plant scan completed — {classification}", classification)
    else:
        insert_log("Detection", "Live plant scan completed — plant health uncertain", "Unknown")

    if auto:
        if classification.startswith("Sick"):
            insert_log(
                "Actuation",
                "Auto fertilizer dosing triggered after plant stress was detected",
                "Fertilized"
            )
            st.session_state["relay_status"]["fertilizer"] = True
        if moisture_value is not None and moisture_value < 45:
            insert_log(
                "Actuation",
                "Auto irrigation triggered — moisture below threshold",
                "Watered"
            )
            st.session_state["relay_status"]["pump"] = True

    return frame, status, classification, score


def get_serial_port_from_secrets(default_port: str = "COM3") -> str:
    """Return the configured serial port from Streamlit secrets or a default."""
    if hasattr(st, "secrets") and st.secrets:
        return st.secrets.get("arduino_serial_port") or st.secrets.get("arduino_port") or default_port
    return default_port


MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_model.h5")
CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "class_indices.json")


def get_local_moisture_value(throttle_seconds: float = 3.0) -> int:
    """Read live soil moisture from the local Arduino serial port (throttled).

    Prevents repeated serial reads on every Streamlit rerun.
    """
    now = time.time()
    last_ts = st.session_state.get("_moisture_last_read_ts", 0)
    if (now - last_ts) < throttle_seconds and "_moisture_last_value" in st.session_state:
        return st.session_state.get("_moisture_last_value")

    port = get_serial_port_from_secrets()
    value = read_arduino_serial_value(port=port)

    st.session_state["_moisture_last_read_ts"] = now
    st.session_state["_moisture_last_value"] = value
    return value



def get_training_class_labels() -> List[str]:
    """Return class labels derived from a saved class index file or local folders."""
    if os.path.exists(CLASS_INDEX_PATH):
        try:
            with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as fh:
                labels = json.load(fh)
            if isinstance(labels, list):
                return labels
        except Exception:
            pass

    base_dir = os.path.dirname(__file__)
    try:
        labels = [
            name for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and "___" in name
        ]
        return sorted(labels)
    except Exception:
        return []


def load_plant_model():
    """Load the trained plant model if available and return the model with class labels."""
    if hasattr(load_plant_model, "_cache"):
        return load_plant_model._cache

    try:
        if not os.path.exists(MODEL_PATH):
            load_plant_model._cache = (None, [])
            return load_plant_model._cache

        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        class_labels = get_training_class_labels()
        load_plant_model._cache = (model, class_labels)
    except Exception:
        load_plant_model._cache = (None, [])
    return load_plant_model._cache


def predict_plant_disease(frame: np.ndarray):
    """Return the model-predicted disease class and confidence if model is available."""
    model, class_labels = load_plant_model()
    if model is None or frame is None or frame.size == 0:
        return None, None

    try:
        rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        batch = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
        preds = model.predict(batch, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        label = class_labels[best_idx] if best_idx < len(class_labels) else f"Class {best_idx}"
        score = int(round(float(preds[best_idx]) * 100))
        return label, score
    except Exception:
        return None, None


def analyze_plant_health(frame: np.ndarray):
    """Return a plant health classification and confidence using model inference or heuristic fallback."""
    if frame is None or frame.size == 0:
        return "Unknown", 0

    label, score = predict_plant_disease(frame)
    if label is not None:
        normalized = label.replace("___", " ").replace("_", " ")
        if "healthy" in label.lower():
            return "Healthy", score
        return f"Sick — {normalized.title()}", score

    # Heuristic fallback when model is not available.
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    brown_mask = cv2.inRange(hsv, (5, 40, 40), (25, 255, 200))

    green_ratio = np.count_nonzero(green_mask) / (frame.shape[0] * frame.shape[1])
    brown_ratio = np.count_nonzero(brown_mask) / (frame.shape[0] * frame.shape[1])
    saturation = np.mean(hsv[:, :, 1]) / 255.0

    if green_ratio > 0.28 and brown_ratio < 0.08 and saturation > 0.35:
        return "Healthy", int(min(98, max(70, green_ratio * 140)))

    severity = int(min(96, max(55, (brown_ratio + (0.35 - saturation)) * 180)))
    return "Sick", severity

GLOBAL_CSS = """
<style>
:root {
  --bg: #F5FBF3;
  --surface: #FFFFFF;
  --surface-soft: #F0F7EE;
  --text: #1E4F28;
  --muted: #4C7D51;
  --accent: #4C8F3F;
  --accent-soft: #D7ECD7;
  --border: rgba(76, 143, 63, 0.18);
  --radius: 20px;
  --shadow: 0 18px 45px rgba(34, 84, 29, 0.08);
}

body, .main, .block-container {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.page-header {
  font-size: 2.4rem;
  font-weight: 800;
  color: #1E4F28;
  margin-bottom: 0.35rem;
}

.page-subheader {
  color: #4C7D51;
  font-size: 1rem;
  margin-bottom: 1.4rem;
  line-height: 1.6;
}

.section-label {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  font-size: 0.9rem;
  font-weight: 700;
  color: #3F6F3F;
  margin-bottom: 0.9rem;
}

.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, #BEE6C8, transparent);
}

.ss-card {
  background: rgba(255,255,255,0.97);
  border-radius: 22px;
  padding: 1.3rem;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  margin-bottom: 1rem;
}

.kpi-card {
  padding: 1rem;
  border-radius: 20px;
  background: linear-gradient(180deg, #F5FBF3 0%, #E8F4E0 100%);
  box-shadow: 0 12px 30px rgba(33,103,43,0.08);
  min-height: 170px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.kpi-icon { font-size: 1.5rem; }
.kpi-value { font-size: 2rem; font-weight: 700; }
.kpi-label { color: #2E6B33; font-weight: 700; margin-top: 0.35rem; }

.stButton > button {
  background: linear-gradient(135deg, #4C8F3F 0%, #2C5F1B 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 999px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  padding: 0.85rem 1.5rem !important;
  box-shadow: 0 12px 28px rgba(36, 93, 34, 0.2) !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
}

.stDataFrame {
  border-radius: 18px !important;
  overflow: hidden !important;
  box-shadow: 0 14px 35px rgba(31, 89, 32, 0.08) !important;
}

.stDataFrame thead tr th {
  background: #E7F3E8 !important;
  color: #4C7D51 !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
}

.webcam-frame {
  background: #F1F8F2;
  border-radius: 22px;
  padding: 0.7rem;
  box-shadow: 0 15px 35px rgba(35, 90, 34, 0.08);
  border: 1px solid #C9E5D0;
}

.weather-tile {
  background: linear-gradient(135deg, #F4FBF5 0%, #DCF1D9 100%);
  border-radius: 20px;
  padding: 1.4rem;
  text-align: center;
  box-shadow: 0 12px 28px rgba(37, 90, 36, 0.08);
  border: 1px solid #C8E7C7;
}

.task-item {
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
  padding: 0.8rem 0;
  border-bottom: 1px solid #D8E8D7;
}
.task-item:last-child { border-bottom: none; }
.task-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  margin-top: 0.4rem;
}

.status-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 0.4rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.status-online  { background: #66BB6A; }
.status-offline { background: #EF5350; }
.status-standby { background: #FFA726; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE LAYER (MySQL with SQLite fallback)
# ─────────────────────────────────────────────

def get_db_connection():
    """Return a DB connection.

    Tries MySQL only if Streamlit secrets are configured; otherwise falls back to local SQLite.
    """
    # 1) Try MySQL (optional)
    try:
        if hasattr(st, "secrets") and st.secrets:
            mysql_cfg = st.secrets.get("mysql", {})
        else:
            mysql_cfg = {}

        # Expect secrets like:
        # mysql: { host: ..., user: ..., password: ..., database: ... }
        if mysql_cfg and mysql_cfg.get("host") and mysql_cfg.get("user") and mysql_cfg.get("password"):
            conn = mysql.connector.connect(
                host=mysql_cfg.get("host"),
                user=mysql_cfg.get("user"),
                password=mysql_cfg.get("password"),
                database=mysql_cfg.get("database"),
            )
            return conn, "mysql"
    except Exception:
        # Ignore and fall back to SQLite
        pass

    # 2) SQLite fallback
    import sqlite3
    conn = sqlite3.connect("sproutsmart.db", check_same_thread=False)
    return conn, "sqlite"



def init_db():
    """Initialize the activity log table."""
    if "db_initialized" not in st.session_state:
        conn, mode = get_db_connection()
        try:
            cur = conn.cursor()
            if mode == "mysql":
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        activity_type VARCHAR(20) NOT NULL,
                        event_description VARCHAR(255),
                        metric_label VARCHAR(30),
                        extra_value FLOAT
                    )
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        activity_type TEXT NOT NULL,
                        event_description TEXT,
                        metric_label TEXT,
                        extra_value REAL
                    )
                """)
            conn.commit()
            # Seed demo data if empty
            cur.execute("SELECT COUNT(*) FROM activity_log")
            count = cur.fetchone()[0]
            if count == 0:
                _seed_demo_data(conn, cur, mode)
            conn.commit()
        finally:
            conn.close()
        st.session_state["db_initialized"] = True
        st.session_state["db_mode"] = mode


def _seed_demo_data(conn, cur, mode):
    """Insert 15 rows of representative demo data."""
    import random
    labels = [
        ("Detection", "Leaf scan completed — plant identified as healthy", "Healthy"),
        ("Detection", "Anomaly detected — potential fungal lesion spotted", "Sick"),
        ("Actuation", "Water pump relay triggered — irrigation cycle started", "Watered"),
        ("Actuation", "Fertilizer relay triggered — nutrient dose dispensed", "Fertilized"),
        ("Detection", "Leaf scan completed — chlorophyll levels normal", "Healthy"),
        ("Detection", "Yellowing detected — possible nitrogen deficiency", "Sick"),
        ("Actuation", "Water pump relay triggered — drought stress response", "Watered"),
        ("Detection", "Leaf scan completed — growth stage nominal", "Healthy"),
        ("Actuation", "Fertilizer relay triggered — scheduled dose applied", "Fertilized"),
        ("Detection", "Brown spots detected — monitoring for spread", "Sick"),
        ("Actuation", "Water pump relay triggered — temperature spike response", "Watered"),
        ("Detection", "Leaf scan completed — all parameters within range", "Healthy"),
        ("Detection", "Wilting detected — soil moisture critically low", "Sick"),
        ("Actuation", "Water pump relay triggered — emergency irrigation", "Watered"),
        ("Detection", "Leaf scan completed — recovery confirmed post-irrigation", "Healthy"),
    ]
    base = datetime.datetime.now() - datetime.timedelta(hours=48)
    for i, (atype, desc, label) in enumerate(labels):
        ts = base + datetime.timedelta(hours=i * 3, minutes=random.randint(0, 59))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        if mode == "mysql":
            cur.execute(
                "INSERT INTO activity_log (timestamp, activity_type, event_description, metric_label) VALUES (%s, %s, %s, %s)",
                (ts_str, atype, desc, label)
            )
        else:
            cur.execute(
                "INSERT INTO activity_log (timestamp, activity_type, event_description, metric_label) VALUES (?, ?, ?, ?)",
                (ts_str, atype, desc, label)
            )


def fetch_logs(cache_seconds: float = 5.0) -> pd.DataFrame:
    """Retrieve all activity log rows as a DataFrame.

    Cached briefly in Streamlit session state to reduce DB load on reruns.
    """
    now = time.time()
    last_ts = st.session_state.get("_logs_last_fetch_ts", 0)
    if (now - last_ts) < cache_seconds and "_logs_cache_df" in st.session_state:
        return st.session_state["_logs_cache_df"]

    conn, mode = get_db_connection()
    try:
        if mode == "mysql":
            df = pd.read_sql("SELECT * FROM activity_log ORDER BY timestamp DESC", conn)
        else:
            df = pd.read_sql_query("SELECT * FROM activity_log ORDER BY timestamp DESC", conn)
        st.session_state["_logs_cache_df"] = df
        st.session_state["_logs_last_fetch_ts"] = now
        return df
    finally:
        conn.close()



def insert_log(activity_type: str, description: str, metric_label: str, extra_value: float = None):
    """Append a new row to the activity log."""
    conn, mode = get_db_connection()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur = conn.cursor()
        if mode == "mysql":
            cur.execute(
                "INSERT INTO activity_log (timestamp, activity_type, event_description, metric_label, extra_value) VALUES (%s,%s,%s,%s,%s)",
                (ts, activity_type, description, metric_label, extra_value)
            )
        else:
            cur.execute(
                "INSERT INTO activity_log (timestamp, activity_type, event_description, metric_label, extra_value) VALUES (?,?,?,?,?)",
                (ts, activity_type, description, metric_label, extra_value)
            )
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────
if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = False
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "🏡 Home Garden"
if "sidebar_nav" not in st.session_state:
    st.session_state["sidebar_nav"] = "🏡 Home Garden"
if "cam_running" not in st.session_state:
    st.session_state["cam_running"] = False
if "last_capture" not in st.session_state:
    st.session_state["last_capture"] = None
if "relay_status" not in st.session_state:
    st.session_state["relay_status"] = {"pump": False, "fertilizer": False}

# ─────────────────────────────────────────────
# LOTTIE / ANIMATION UTILITIES
# ─────────────────────────────────────────────


def render_plant_animation_fallback():
    """Pure CSS animated plant icon when Lottie is unavailable."""
    return """
    <div style="display:flex;justify-content:center;margin:0.5rem 0;">
      <div style="font-size:3rem;animation:bounce 2s ease-in-out infinite;">🌱</div>
    </div>
    <style>
    @keyframes bounce {
      0%,100%{transform:translateY(0);}
      50%{transform:translateY(-8px);}
    }
    </style>
    """


def render_splash_screen():
    """Display the startup splash screen before the main app loads."""
    st.markdown("""
    <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;
                background: linear-gradient(160deg, #EAF8EB 0%, #F8FFF8 50%, #DFF0D8 100%);padding:1rem;">
      <div style="max-width:760px;width:100%;background:#FFFFFF;border-radius:26px;
                  box-shadow:0 24px 60px rgba(29, 86, 34, 0.12);padding:3rem 2rem;text-align:center;">
        <div style="font-size:4rem;">🌿</div>
        <div style="font-size:2.6rem;font-weight:800;color:#1E4F28;margin-top:0.8rem;">
          SproutSmart
        </div>
        <div style="font-size:1rem;color:#4C7D51;margin-top:0.6rem;letter-spacing:0.02em;">
          A simple plant care dashboard for your farm.
        </div>
        <div style="display:grid;grid-template-columns:repeat(3, minmax(0,1fr));gap:0.85rem;margin-top:2rem;">
          <div style="background:#F6FBF5;border-radius:18px;padding:1rem;font-weight:700;color:#2F6C33;">📸 Quick Scan</div>
          <div style="background:#F6FBF5;border-radius:18px;padding:1rem;font-weight:700;color:#2F6C33;">💧 Moisture</div>
          <div style="background:#F6FBF5;border-radius:18px;padding:1rem;font-weight:700;color:#2F6C33;">📊 Farm Log</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start SproutSmart", use_container_width=True):
        st.session_state["app_loaded"] = True
        st.rerun()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION WITH LOGS
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0.8rem 0.6rem;">
          <div style="font-family:'DM Serif Display',serif;font-size:1.7rem;color:#1B5E20;letter-spacing:-0.01em;">
            🌿 SproutSmart
          </div>
          <div style="font-size:0.78rem;color:#2E7D32;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.25rem;">
            Friendly crop care for busy growers
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        uname = st.session_state.get("username", "Farmer")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.95);border-radius:18px;padding:1rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.8rem;">
          <div style="width:42px;height:42px;border-radius:50%;background:#66BB6A;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem;color:#ffffff;">{uname[0].upper() if uname else 'F'}</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:1rem;color:#1B5E20;">Hello, {uname.capitalize()}!</div>
            <div style="font-size:0.74rem;color:#4A7D4A;line-height:1.4;">Your garden helper is ready.</div>
          </div>
          <div style="padding:0.35rem 0.75rem;border-radius:999px;background:#E8F5E9;color:#2E7D32;font-size:0.72rem;font-weight:700;">Live</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#4A7D4A;margin-bottom:0.5rem;padding-left:0.2rem;">Tap around</div>
        """, unsafe_allow_html=True)

        view = st.radio(
            "Navigation",
            ["🏡 Home Garden", "📸 Quick Scan", "🌤️ Weather Check", "📚 Farm History"],
            key="sidebar_nav",
            label_visibility="collapsed"
        )
        st.session_state["current_view"] = view

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:#388E3C;margin-bottom:0.6rem;padding-left:0.2rem;">
          🌼 Your favorite crops
        </div>
        """, unsafe_allow_html=True)

        plants = ["Tomato 🍅", "Potato 🥔", "Corn 🌽", "Grape 🍇", "Apple 🍎"]
        for plant in plants:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,0.95);padding:0.7rem 0.9rem;border-radius:14px;margin-bottom:0.45rem;color:#1B5E20;">
              <div style="font-size:0.9rem;font-weight:600;">{plant}</div>
              <div style="font-size:0.7rem;color:#4A7D4A;">Healthy</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#4A7D4A;margin-bottom:0.6rem;padding-left:0.2rem;">
          🌟 Quick tips
        </div>
        <div style="font-size:0.82rem;color:#355E35;line-height:1.55;background:rgba(232,245,233,0.9);padding:0.9rem;border-radius:14px;">
          Upload a plant photo on the home screen, check moisture levels, and let Sprout guide your next steps.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem;color:#4A7D4A;text-align:center;margin-top:1rem;padding:0 0.6rem;">
          SproutSmart © {year}<br><span style="color:#2E7D32;">Made for simple farm care</span>
        </div>
        """.format(year=datetime.datetime.now().year), unsafe_allow_html=True)

    return view


# ─────────────────────────────────────────────
# VIEW A: HOME OVERVIEW
# ─────────────────────────────────────────────
def view_home():
    st.title("Home Garden")
    soil_moisture = get_local_moisture_value()
    moisture_display = f"{soil_moisture}%" if soil_moisture is not None else "Unavailable"
    moisture_tip = (
        "Soil moist and happy. Keep the roots cozy." if soil_moisture is None or soil_moisture >= 65
        else "Soil is dry. Consider irrigating soon."
    )

    # ── Header ──
    col_title, col_anim = st.columns([3, 1])
    with col_title:
        now = datetime.datetime.now()
        hour = now.hour
        greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
        uname = st.session_state.get("username", "Farmer").capitalize()

        st.markdown(f"""
        <div class="page-header">{greeting}, {uname}! 🌱</div>
        <div class="page-subheader">
          {now.strftime('%A, %B %d %Y')} · A cozy plant care dashboard for your crops.
        </div>
        """, unsafe_allow_html=True)
    with col_anim:
        st.markdown(render_plant_animation_fallback(), unsafe_allow_html=True)

    # ── Live Moisture (Arduino → data_collector.py) ──
    st.markdown('<div class="section-label">💧 Live Soil Moisture</div>', unsafe_allow_html=True)
    moisture_col, status_col = st.columns([2, 1])
    with moisture_col:
        placeholder = st.empty()
        refresh_ms = st.session_state.get("moisture_refresh_ms", 1500)
        refresh_ms = st.slider(
            "Update interval (ms)",
            min_value=500,
            max_value=5000,
            value=int(refresh_ms),
            step=100,
            key="moisture_refresh_ms_ui",
        )
        st.session_state["moisture_refresh_ms"] = refresh_ms

        last_refresh = st.session_state.get("_moisture_last_refresh_ts", 0.0)
        now = time.time()

        if now - last_refresh >= (refresh_ms / 1000.0):
            st.session_state["_moisture_last_refresh_ts"] = now
            payload = get_latest_moisture_value() if get_latest_moisture_value else None
            if payload is None:
                placeholder.error("Moisture collector not available. Start: python data_collector.py")
            else:
                if payload.connected and payload.moisture_pct is not None:
                    placeholder.success(f"{payload.moisture_pct}%")
                else:
                    if payload.error:
                        placeholder.warning(f"No data yet: {payload.error}")
                    else:
                        placeholder.warning("Waiting for Arduino data…")

        # trigger rerun without manual browser refresh
        # Streamlit reruns the script automatically; using st.rerun() only when interval elapsed.
        if now - st.session_state.get("_moisture_last_refresh_ts", now) < (refresh_ms / 1000.0):
            pass

        # Always display a small details line
        if get_latest_moisture_value:
            payload = get_latest_moisture_value()
            if payload and payload.connected:
                ts = payload.timestamp_iso[:19] if payload.timestamp_iso else "—"
                placeholder.caption(f"Last update (UTC): {ts}")
            elif payload:
                placeholder.caption(f"Disconnected: {payload.error or 'unknown error'}")

    with status_col:
        if get_latest_moisture_value:
            payload = get_latest_moisture_value()
            if payload and payload.connected:
                relay_state = "ON" if payload.relay_on else "OFF"
                st.markdown(
                    f"""
                    <div class="ss-card" style="padding:1rem 1.1rem;border-left:4px solid #42A5F5;">
                      <div style="font-size:0.78rem;font-weight:800;color:#4A6741;text-transform:uppercase;letter-spacing:0.08em;">Relay</div>
                      <div style="font-size:1.6rem;font-weight:800;color:#1565C0;margin-top:0.35rem;">{relay_state}</div>
                      <div style="font-size:0.7rem;color:#4A6741;margin-top:0.35rem;">Raw: {payload.raw if payload.raw is not None else "—"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                err = payload.error if payload else "collector_not_started"
                st.info(f"Arduino status: Disconnected\n({err})")
        else:
            st.info("Start data_collector.py to enable live Arduino readings.")

    # Rerun loop for live updates
    if "_moisture_last_refresh_ts" in st.session_state:
        # Use st.experimental_rerun to avoid waiting too long; only rerun when the interval has elapsed.
        now2 = time.time()
        if (now2 - st.session_state.get("_moisture_last_refresh_ts", now2)) >= (st.session_state.get("moisture_refresh_ms", 1500) / 1000.0):
            st.rerun()

    # ── Live Demo Upload ──
    st.markdown('<div class="section-label">📸 Snap a leaf</div>', unsafe_allow_html=True)

    demo_col, result_col = st.columns([2, 1])
    with demo_col:
        uploaded_file = st.file_uploader("Take a photo of a plant leaf and see a friendly health summary", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            np_image = np.array(image)
            classification, score = analyze_plant_health(np_image)
            message = (
                "Your plant looks happy! Keep watering regularly." if classification == "Healthy"
                else "A little care will help it bounce back. Consider a close inspection."
            )
            st.image(image, caption="Plant snapshot", width=640)
            st.markdown(f"""
            <div class="ss-card" style="border-left:4px solid #66BB6A;">
              <div style="font-size:0.95rem;font-weight:700;color:#1E5631;">Sprout says</div>
              <div style="font-size:1.6rem;color:#1B5E20;margin-top:0.5rem;">{classification}</div>
              <div style="font-size:0.9rem;color:#3C6D43;margin-top:0.35rem;">Confidence: {score}%</div>
              <div style="margin-top:1rem;color:#2E6B35;line-height:1.5;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload a leaf photo to get a gentle plant health check.")

    with result_col:
        st.markdown(f"""
        <div class="ss-card" style="border-left:4px solid #64B5F6;">
          <div style="font-size:0.85rem;font-weight:700;color:#1B5E20;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.8rem;">
            Soil sweetness
          </div>
          <div style="font-size:2.4rem;font-family:'DM Serif Display',serif;color:#1B5E20;">{moisture_display}</div>
          <div style="font-size:0.88rem;color:#3C6D43;margin-top:0.5rem;line-height:1.45;">
            Soil moisture is based on your local Arduino sensor.
          </div>
          <div style="margin-top:1rem;background:#E8F5E9;border-radius:14px;padding:0.95rem;color:#2E6B35;font-size:0.85rem;">
            {moisture_tip}
          </div>
        </div>
        """, unsafe_allow_html=True)

    df = fetch_logs()
    healthy_count   = int((df["metric_label"] == "Healthy").sum())   if not df.empty else 0
    sick_count      = int((df["metric_label"] == "Sick").sum())      if not df.empty else 0
    watered_count   = int((df["metric_label"] == "Watered").sum() + (df["metric_label"] == "Fertilized").sum()) if not df.empty else 0

    st.markdown('<div class="section-label">📈 Garden tracker</div>', unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:#66BB6A;">
          <div class="kpi-icon">🟩</div>
          <div class="kpi-value" style="color:#2E7D32;">{healthy_count}</div>
          <div class="kpi-label">Healthy Scans</div>
          <div style="font-size:0.75rem;color:#81C784;margin-top:0.4rem;">All-time detections</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:#E57373;">
          <div class="kpi-icon">🟥</div>
          <div class="kpi-value" style="color:#C62828;">{sick_count}</div>
          <div class="kpi-label">Sick Scans Detected</div>
          <div style="font-size:0.75rem;color:#E57373;margin-top:0.4rem;">Requires attention</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:#64B5F6;">
          <div class="kpi-icon">💧</div>
          <div class="kpi-value" style="color:#1565C0;">{watered_count}</div>
          <div class="kpi-label">Relay Spray Cycles</div>
          <div style="font-size:0.75rem;color:#64B5F6;margin-top:0.4rem;">Total actuations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Two-column lower section ──
    col_tasks, col_env = st.columns([1.3, 1])

    with col_tasks:
        st.markdown("### 📋 Today's reminders")

        tasks = [
            ("🌱", "Happy plant check", "Upload a leaf photo to keep a close watch on crop health"),
            ("💧", "Water reminder", "Soil looks good now — keep the drip timer on rhythm"),
            ("☀️", "Sunlight note", "Bright sunlight is ideal, enjoy the warm growing day"),
            ("🛠️", "Tune-up", "Visit the hardware panel to keep relays and sensors humming"),
        ]

        for icon, priority, msg in tasks:
            task_cols = st.columns([0.12, 0.7, 0.18])
            with task_cols[0]:
                st.markdown(f"<div style='font-size:1.4rem;text-align:center;'>{icon}</div>", unsafe_allow_html=True)
            with task_cols[1]:
                st.markdown(f"""**{priority}**  
{msg}""", unsafe_allow_html=True)
            with task_cols[2]:
                if st.button("Done", key=f"task_done_{priority}"):
                    st.success(f"{priority} completed")

    with col_env:
        st.markdown('<div class="section-label">🌿 Garden conditions</div>', unsafe_allow_html=True)

        env_data = [
            ("🌡️", "Temperature",    "28°C",   "Warm and comfy for crops",      "#66BB6A"),
            ("💧", "Soil Moisture",  "67%",    "Happy and hydrated",           "#42A5F5"),
            ("☀️", "Sunlight",       "Bright", "Great growing light",         "#FFA726"),
            ("🍃", "Leaf Health",    "98%",    "Greener than yesterday",       "#66BB6A"),
        ]

        for icon, label, value, note, color in env_data:
            row = st.columns([0.12, 0.6, 0.28])
            with row[0]:
                st.markdown(f"<div style='font-size:1.3rem;text-align:center;'>{icon}</div>", unsafe_allow_html=True)
            with row[1]:
                st.markdown(f"""**{label}**  
{note}""", unsafe_allow_html=True)
            with row[2]:
                st.markdown(f"<div style='font-family:DM Serif Display, serif; font-size:1.1rem; color:{color}; font-weight:600;'>{value}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # ── Recent activity mini-feed ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🕐 Plant care log</div>', unsafe_allow_html=True)

    if not df.empty:
        recent = df.head(5)
        cols = st.columns(5)
        for i, (_, row) in enumerate(recent.iterrows()):
            with cols[i]:
                label = row.get("metric_label", "—")
                color_map = {"Healthy":"#66BB6A","Sick":"#EF5350","Watered":"#42A5F5","Fertilized":"#AB47BC"}
                color = color_map.get(label, "#81C784")
                ts = str(row.get("timestamp", ""))[:16]
                desc = str(row.get("event_description", ""))[:55] + "…"
                st.markdown(f"""
                <div style="background:white;border-radius:12px;padding:1rem;
                            box-shadow:0 2px 8px rgba(56,142,60,0.08);
                            border-top:3px solid {color};height:100%;">
                  <div style="font-size:0.7rem;font-weight:700;color:{color};
                              text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
                  <div style="font-size:0.75rem;color:#4A6741;margin:0.4rem 0;line-height:1.4;">{desc}</div>
                  <div style="font-size:0.68rem;color:#8BAD87;">{ts}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No activity logs yet — start scanning to populate the feed.")


# ─────────────────────────────────────────────
# VIEW B: HARDWARE CONTROL PANEL
# ─────────────────────────────────────────────
def view_hardware():
    """Render the hardware control panel with webcam scanning and relay controls."""
    st.markdown('<div class="page-header">⚙️ Hardware Control Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Webcam monitoring, relay actuation, and system controls</div>', unsafe_allow_html=True)

    col_cam, col_ctrl = st.columns([1.4, 1])

    # ── Left: Webcam and scan controls ──
    with col_cam:
        st.markdown('<div class="section-label">📷 Live Webcam Monitor</div>', unsafe_allow_html=True)

        camera_index = st.number_input(
            "Webcam index",
            min_value=0,
            max_value=4,
            value=st.session_state.get("camera_index", 0),
            step=1,
            key="camera_index"
        )

        moisture_value = get_local_moisture_value()
        moisture_text = (
            f"Live soil moisture: {moisture_value}%" if moisture_value is not None
            else "Live soil moisture unavailable. Check Arduino serial connection."
        )

        with st.container():
            st.markdown("""
            <div class="webcam-frame" style="min-height:280px;display:flex;
                 align-items:center;justify-content:center;">
            """, unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶  Capture & Scan Now", use_container_width=True):
                    frame, status, classification, score = perform_scan(camera_index, moisture_value, auto=False)
                    st.session_state["last_capture"] = frame
                    st.session_state["last_capture_time"] = datetime.datetime.now().strftime("%H:%M:%S")
                    st.session_state["last_scan_classification"] = classification
                    st.session_state["last_scan_score"] = score
                    st.session_state["last_scan_status"] = status
                    if frame is None:
                        st.warning("⚠️ Webcam feed not available. Please verify camera connection.")
                    elif classification == "Healthy":
                        st.success(f"✅ Scan complete — {classification} ({score}%)")
                    else:
                        st.warning(f"⚠️ Scan complete — {classification} ({score}%)")

            with col_btn2:
                if st.button("🔄  Toggle Auto Scan", key="toggle_auto_scan", use_container_width=True):
                    st.session_state["cam_running"] = not st.session_state.get("cam_running", False)
                    if st.session_state["cam_running"]:
                        st.success("Auto scan enabled")
                    else:
                        st.info("Auto scan disabled")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.92rem;color:#4A6741;margin-top:0.8rem;'>{moisture_text}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # Display last captured frame
            if st.session_state.get("last_capture") is not None:
                ts_label = st.session_state.get("last_capture_time", "")
                st.image(
                    st.session_state["last_capture"],
                    caption=f"Last capture at {ts_label}",
                    use_container_width=True
                )
            else:
                st.markdown("""
                <div style="background:#0D1F0F;border-radius:10px;height:220px;
                     display:flex;align-items:center;justify-content:center;
                     border:2px dashed #81C784;">
                  <div style="text-align:center;color:#81C784;">
                    <div style="font-size:2rem;">📷</div>
                    <div style="font-size:0.8rem;margin-top:0.5rem;">
                      Press 'Capture & Scan Now' to start<br>webcam monitoring
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Auto-scan scheduling and capture logic
        is_running = st.session_state.get("cam_running", False)
        interval_sec = st.session_state.get("polling_interval_sec", 300)
        last_auto = st.session_state.get("last_auto_scan", 0)
        now = time.time()
        next_scan = last_auto + interval_sec

        if is_running and now >= next_scan:
            frame, status, classification, score = perform_scan(camera_index, moisture_value, auto=True)
            st.session_state["last_capture"] = frame
            st.session_state["last_capture_time"] = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state["last_scan_classification"] = classification
            st.session_state["last_scan_score"] = score
            st.session_state["last_scan_status"] = status
            st.session_state["last_auto_scan"] = now
            if frame is None:
                st.warning("⚠️ Auto scan attempted, but webcam could not be read.")
            elif classification == "Healthy":
                st.success(f"✅ Auto scan complete — {classification} ({score}%)")
            else:
                st.warning(f"⚠️ Auto scan complete — {classification} ({score}%)")

        if is_running:
            time_left = max(0, int(next_scan - now))
            st.markdown(f"<div style='margin-top:0.35rem;color:#4A7D1A;'>Auto scan is active. Next scan in {time_left}s.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top:0.35rem;color:#8BAD87;'>Auto scan is currently paused.</div>", unsafe_allow_html=True)

# ── Right: Control Panel ──
    with col_ctrl:
        st.markdown('<div class="section-label">🔧 Garden controls</div>', unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        # Non-blocking irrigation controller (Arduino-like)
        # ─────────────────────────────────────────────
        # Tested Arduino logic:
        # - Pump ON when soil moisture percentage < triggerThreshold
        # - Run pump for waterTime (ms)
        # - Wait soakTime (ms)
        # - Re-check moisture% and potentially run again
        #
        # This controller is time-driven (no delay()), so the UI stays responsive.

        # Defaults (derived from your Arduino sketch)
        triggerThreshold_pct = 30
        waterTime_ms = 2000
        soakTime_ms = 5000

        st.session_state.setdefault("auto_irrigation_enabled", True)
        st.session_state.setdefault("irrigation_state", "IDLE")  # IDLE, PUMPING, SOAKING
        st.session_state.setdefault("irrigation_next_check_ts", 0.0)
        st.session_state.setdefault("irrigation_cycle_started_ts", None)
        st.session_state.setdefault("irrigation_last_watered_ts", None)

        # UI inputs: allow user to adjust thresholds/timings
        st.subheader("🚿 Auto Irrigation (Non-blocking)")
        st.session_state["auto_irrigation_enabled"] = st.toggle(
            "Enable auto irrigation",
            value=st.session_state["auto_irrigation_enabled"],
            key="auto_irrigation_toggle",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            triggerThreshold_pct = st.slider(
                "Trigger below (%)",
                min_value=0,
                max_value=100,
                value=triggerThreshold_pct,
            )
        with col_b:
            water_s = st.slider(
                "Pump time (seconds)",
                min_value=1,
                max_value=10,
                value=int(round(waterTime_ms / 1000)),
            )
        soak_s = st.slider(
            "Soak time (seconds)",
            min_value=1,
            max_value=20,
            value=int(round(soakTime_ms / 1000)),
        )

        waterTime_ms = int(water_s * 1000)
        soakTime_ms = int(soak_s * 1000)

        # Convert raw Arduino reading (0-1023) -> percentage like your Arduino map()
        dryValue = 500
        wetValue = 200
        moisture_value_raw = moisture_value  # from above UI
        if moisture_value_raw is None:
            moisture_pct = None
        else:
            # map(sensorValue, dryValue, wetValue, 0, 100) and constrain
            moisture_pct = int((moisture_value_raw - dryValue) * (0 - 100) / (wetValue - dryValue))
            moisture_pct = max(0, min(100, moisture_pct))

        now_ts = time.time()

        # State machine update
        if st.session_state["auto_irrigation_enabled"]:
            if st.session_state["irrigation_state"] == "IDLE":
                if moisture_pct is not None and moisture_pct < triggerThreshold_pct:
                    # Start pump
                    st.session_state["irrigation_state"] = "PUMPING"
                    st.session_state["irrigation_cycle_started_ts"] = now_ts
                    st.session_state["irrigation_next_check_ts"] = now_ts + (waterTime_ms / 1000.0)

                    st.session_state["relay_status"]["pump"] = True
                    st.session_state["relay_status"]["fertilizer"] = st.session_state["relay_status"].get("fertilizer", False)

                    insert_log(
                        "Actuation",
                        f"Auto irrigation started — moisture {moisture_pct}% < {triggerThreshold_pct}%",
                        "Watered",
                    )

            elif st.session_state["irrigation_state"] == "PUMPING":
                if now_ts >= st.session_state["irrigation_next_check_ts"]:
                    # Stop pump and begin soak
                    st.session_state["irrigation_state"] = "SOAKING"
                    st.session_state["irrigation_next_check_ts"] = now_ts + (soakTime_ms / 1000.0)
                    st.session_state["relay_status"]["pump"] = False
                    st.session_state["irrigation_last_watered_ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    insert_log(
                        "Actuation",
                        "Auto irrigation pump stopped — soaking period",
                        "Watered",
                    )

            elif st.session_state["irrigation_state"] == "SOAKING":
                if now_ts >= st.session_state["irrigation_next_check_ts"]:
                    # Re-check moisture and either start another cycle or go idle
                    # (Moisture re-check occurs automatically on next rerun via get_local_moisture_value())
                    if moisture_pct is not None and moisture_pct < triggerThreshold_pct:
                        st.session_state["irrigation_state"] = "PUMPING"
                        st.session_state["irrigation_cycle_started_ts"] = now_ts
                        st.session_state["irrigation_next_check_ts"] = now_ts + (waterTime_ms / 1000.0)

                        st.session_state["relay_status"]["pump"] = True
                        st.session_state["irrigation_last_watered_ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        insert_log(
                            "Actuation",
                            f"Auto irrigation re-cycle — moisture {moisture_pct}% still < {triggerThreshold_pct}%",
                            "Watered",
                        )
                    else:
                        st.session_state["irrigation_state"] = "IDLE"
                        st.session_state["relay_status"]["pump"] = False
        else:
            # If disabled, force IDLE and pump OFF
            st.session_state["irrigation_state"] = "IDLE"
            st.session_state["relay_status"]["pump"] = False

        # UI readout
        state = st.session_state["irrigation_state"]
        pump_cycle_status = "IDLE"
        color = "#8BAD87"
        if state == "PUMPING":
            pump_cycle_status = "PUMPING"
            color = "#42A5F5"
        elif state == "SOAKING":
            pump_cycle_status = "SOAKING"
            color = "#FFA726"

        last_watered = st.session_state.get("irrigation_last_watered_ts")
        remaining = max(0, int(st.session_state.get("irrigation_next_check_ts", now_ts) - now_ts))

        st.markdown(
            f"""
            <div style='background:#F1F8F1;border-radius:14px;padding:0.8rem 1rem;margin-top:0.5rem;border:1px solid rgba(76, 143, 63, 0.18);'>
              <div style='font-size:0.8rem;font-weight:800;color:#4A6741;text-transform:uppercase;letter-spacing:0.08em;'>🚰 Status</div>
              <div style='display:flex;justify-content:space-between;gap:1rem;align-items:center;margin-top:0.4rem;'>
                <div style='font-family:"DM Serif Display",serif;font-size:1.6rem;color:{color};font-weight:700;'>{pump_cycle_status}</div>
                <div style='font-size:0.78rem;color:#4A6741;text-align:right;'>
                  <div>Moisture%: <b>{moisture_pct if moisture_pct is not None else "—"}</b></div>
                  <div>Next in: <b>{remaining}s</b></div>
                  <div>Last watered: <b>{last_watered if last_watered else "—"}</b></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)


        # Water pump
        st.markdown("""
        <div class="ss-card" style="border-left-color:#42A5F5;">
          <div style="font-size:0.7rem;font-weight:700;color:#42A5F5;letter-spacing:0.1em;
                      text-transform:uppercase;margin-bottom:0.5rem;">💧 Water pump</div>
          <div style="font-size:0.82rem;color:#4A6741;margin-bottom:0.8rem;line-height:1.5;">
            Manually triggers the drip irrigation pump. Initiates a 90-second water delivery cycle
            to all connected irrigation zones.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚿  Manual Trigger: Water Pump Relay", use_container_width=True):
            insert_log("Actuation", "Water pump relay manually triggered by operator", "Watered")
            st.session_state["relay_status"]["pump"] = True
            st.success("✅ Water pump relay activated — irrigation cycle started (90s)")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Fertilizer relay
        st.markdown("""
        <div class="ss-card" style="border-left-color:#AB47BC;">
          <div style="font-size:0.7rem;font-weight:700;color:#AB47BC;letter-spacing:0.1em;
                      text-transform:uppercase;margin-bottom:0.5rem;">🌾 Nutrient pump</div>
          <div style="font-size:0.82rem;color:#4A6741;margin-bottom:0.8rem;line-height:1.5;">
            Manually triggers the fertilizer dosing pump. Dispenses a calibrated nutrient solution
            mixed at 1:200 dilution ratio.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧪  Manual Trigger: Fertilizer Relay", use_container_width=True):
            insert_log("Actuation", "Fertilizer relay manually triggered by operator", "Fertilized")
            st.session_state["relay_status"]["fertilizer"] = True
            st.success("✅ Fertilizer relay activated — nutrient dose dispensed")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Relay status readout
        st.markdown('<div class="section-label">🔌 Relay Status</div>', unsafe_allow_html=True)
        pump_st = "ACTIVE" if st.session_state["relay_status"]["pump"] else "IDLE"
        fert_st = "ACTIVE" if st.session_state["relay_status"]["fertilizer"] else "IDLE"
        pump_c  = "#66BB6A" if st.session_state["relay_status"]["pump"] else "#8BAD87"
        fert_c  = "#66BB6A" if st.session_state["relay_status"]["fertilizer"] else "#8BAD87"

        st.markdown(f"""
        <div class="ss-card" style="padding:1rem 1.2rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:0.6rem;">
            <span style="font-size:0.82rem;color:#4A6741;">💧 Water Pump</span>
            <span style="font-size:0.78rem;font-weight:700;color:{pump_c};">{pump_st}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:0.82rem;color:#4A6741;">🌾 Fertilizer</span>
            <span style="font-size:0.78rem;font-weight:700;color:{fert_c};">{fert_st}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Bottom: Polling Interval Slider ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">⏱️ Scan schedule</div>', unsafe_allow_html=True)

    slider_col, info_col = st.columns([2, 1])
    with slider_col:
        interval_mode = st.selectbox(
            "Interval Unit",
            ["Minutes", "Hours"],
            key="interval_mode",
            label_visibility="collapsed"
        )
        if interval_mode == "Minutes":
            interval_val = st.slider(
                "Capture Interval (minutes)", min_value=1, max_value=60,
                value=st.session_state.get("capture_interval_min", 5),
                key="capture_interval_min"
            )
            interval_label = f"{interval_val} minutes"
            interval_sec = interval_val * 60
        else:
            interval_val = st.slider(
                "Capture Interval (hours)", min_value=1, max_value=24,
                value=st.session_state.get("capture_interval_hr", 1),
                key="capture_interval_hr"
            )
            interval_label = f"{interval_val} hour{'s' if interval_val > 1 else ''}"
            interval_sec = interval_val * 3600

        st.session_state["polling_interval_sec"] = interval_sec

    with info_col:
        scans_per_day = round(86400 / interval_sec, 1)
        st.markdown(f"""
        <div class="ss-card" style="padding:1rem;text-align:center;">
          <div style="font-size:0.7rem;color:#8BAD87;text-transform:uppercase;
                      letter-spacing:0.08em;">Polling Interval</div>
          <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;
                      color:#2E7D32;margin:0.3rem 0;">{interval_label}</div>
          <div style="font-size:0.73rem;color:#66BB6A;">≈ {scans_per_day} scans/day</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VIEW C: WEATHER DIAGNOSTICS
# ─────────────────────────────────────────────
def view_weather():
    st.markdown('<div class="page-header">🌤️ Weather Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Real-time microclimate analysis and irrigation risk assessment</div>', unsafe_allow_html=True)

    # City input
    city_col, btn_col = st.columns([3, 1])
    with city_col:
        city = st.text_input(
            "Location",
            value=st.session_state.get("weather_city", "Bengaluru"),
            placeholder="Enter city name (e.g. Bengaluru, Mumbai, Delhi, New York)",
            label_visibility="collapsed",
            key="weather_city_input"
        )
    with btn_col:
        fetch_btn = st.button("🔍  Fetch Climate Data", use_container_width=True)

    if fetch_btn:
        st.session_state["weather_city"] = city.strip() or st.session_state.get("weather_city", "Bengaluru")
        st.session_state["weather_fetched"] = True

    city_to_use = city.strip() or st.session_state.get("weather_city", "Bengaluru")
    st.session_state["weather_city"] = city_to_use

    # Fetch weather using Open-Meteo (free, no API key)
    def get_weather(city_name):
        try:
            # Geocode using request params for safe city names
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {
                "name": city_name,
                "count": 1,
                "language": "en",
                "format": "json",
            }
            geo_r = requests.get(geo_url, params=geo_params, timeout=8)
            geo_data = geo_r.json()
            if "results" not in geo_data or not geo_data["results"]:
                return None, "City not found. Try a simpler city name."
            loc = geo_data["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            country = loc.get("country", "")

            wx_url = "https://api.open-meteo.com/v1/forecast"
            wx_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "hourly": "relativehumidity_2m,precipitation,cloudcover,windspeed_10m,apparent_temperature",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 3,
                "timezone": "auto",
            }
            wx_r = requests.get(wx_url, params=wx_params, timeout=8)
            wx_data = wx_r.json()

            current = wx_data.get("current_weather", {})
            if not current:
                current = {
                    "temperature": wx_data.get("hourly", {}).get("temperature_2m", [None])[0],
                }

            return {
                "city": city_name,
                "country": country,
                "lat": lat,
                "lon": lon,
                "current": {
                    "temperature_2m": current.get("temperature", wx_data.get("hourly", {}).get("temperature_2m", [None])[0]),
                    "relative_humidity_2m": wx_data.get("hourly", {}).get("relativehumidity_2m", [None])[0],
                    "precipitation": wx_data.get("hourly", {}).get("precipitation", [0])[0],
                    "wind_speed_10m": wx_data.get("hourly", {}).get("windspeed_10m", [None])[0],
                    "apparent_temperature": current.get("apparent_temperature", None),
                },
                "daily": wx_data.get("daily", {}),
            }, None
        except Exception as e:
            return None, str(e)

    with st.spinner(f"Fetching climate data for {city_to_use}…"):
        weather, err = get_weather(city_to_use)

    if err:
        st.error(f"⚠️ Could not fetch weather data: {err}")
        # Show demo data
        weather = {
            "city": city_to_use, "country": "India",
            "current": {
                "temperature_2m": 27.4, "relative_humidity_2m": 68,
                "precipitation": 0.2, "wind_speed_10m": 12.3,
                "cloud_cover": 45, "apparent_temperature": 29.1
            },
            "daily": {
                "precipitation_sum": [0.2, 3.4, 1.1],
                "temperature_2m_max": [30, 28, 27],
                "temperature_2m_min": [22, 21, 20],
            }
        }
        st.info("Showing demonstration data — live fetch unavailable")

    if weather:
        cur = weather.get("current", {})
        daily = weather.get("daily", {})

        # Try to override the humidity tile with live Arduino moisture data.
        arduino_note = "Live Arduino soil moisture reading"
        arduino_status = None

        arduino_value = get_local_moisture_value()
        if arduino_value is not None:
            cur["relative_humidity_2m"] = arduino_value
            arduino_status = "LIVE"
        else:
            arduino_note = "Live soil moisture unavailable. Check Arduino serial connection."

        # Location header
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2E7D32 0%,#1B5E20 100%);
                    border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1.2rem;
                    display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                        color:white;letter-spacing:-0.02em;">
              📍 {weather['city']}, {weather['country']}
            </div>
            <div style="font-size:0.8rem;color:#A5D6A7;margin-top:0.2rem;">
              Lat {weather.get('lat','—'):.2f}° · Lon {weather.get('lon','—'):.2f}°
            </div>
          </div>
          <div style="text-align:right;color:white;">
            <div style="font-size:3rem;font-family:'DM Serif Display',serif;">
              {cur.get('temperature_2m','—')}°C
            </div>
            <div style="font-size:0.8rem;color:#C8E6C9;">
              {('Warm' if float(cur.get('temperature_2m', 25)) >= 28 else 'Cool')}
              · Feels like {cur.get('apparent_temperature','—')}°C
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Main metric tiles
        st.markdown('<div class="section-label">📊 Current Conditions</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        tiles = [
            (m1, "🌡️", "Ambient Temperature", f"{cur.get('temperature_2m','—')}°C",
             "Optimal range: 20–35°C for most crops", "#EF5350",
             "OK" if 15 <= float(cur.get('temperature_2m', 25)) <= 38 else "OUT OF RANGE"),
            (m2, "💧", "Relative Humidity", f"{cur.get('relative_humidity_2m','—')}%",
             arduino_note, "#42A5F5",
             arduino_status or ("HIGH" if float(cur.get('relative_humidity_2m', 65)) > 80 else "NORMAL")),
            (m3, "🌧️", "Precipitation", f"{cur.get('precipitation','—')} mm",
             "Current-hour rainfall detected by sensor", "#7E57C2",
             "RAINING" if float(cur.get('precipitation', 0)) > 0 else "CLEAR"),
            (m4, "💨", "Wind Speed", f"{cur.get('wind_speed_10m','—')} km/h",
             "High wind → increased evaporation demand", "#FF7043",
             "HIGH" if float(cur.get('wind_speed_10m', 10)) > 30 else "NORMAL"),
        ]

        for col, icon, label, value, note, color, status in tiles:
            with col:
                st.markdown(f"""
                <div class="weather-tile">
                  <div style="font-size:1.8rem;">{icon}</div>
                  <div class="weather-value">{value}</div>
                  <div style="font-size:0.75rem;font-weight:700;color:{color};
                              text-transform:uppercase;letter-spacing:0.07em;">{label}</div>
                  <div style="font-size:0.7rem;color:#4A6741;margin-top:0.4rem;
                              line-height:1.4;">{note}</div>
                  <div style="margin-top:0.5rem;">
                    <span class="badge" style="background:{color}20;color:{color};">
                      {status}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Irrigation risk assessment
        st.markdown('<div class="section-label">🚿 Irrigation Risk Assessment</div>', unsafe_allow_html=True)

        humidity = float(cur.get('relative_humidity_2m', 65))
        precip   = float(cur.get('precipitation', 0))
        temp     = float(cur.get('temperature_2m', 27))

        risk_score = 0
        if precip > 2:    risk_score -= 40
        elif precip > 0:  risk_score -= 20
        if humidity > 85: risk_score -= 25
        elif humidity > 70: risk_score -= 10
        if temp > 35:     risk_score += 30
        elif temp > 28:   risk_score += 15
        risk_score = max(0, min(100, 50 + risk_score))

        risk_label = "Low Risk — Skip Irrigation" if risk_score < 35 else \
                     "Moderate — Monitor Closely"  if risk_score < 65 else \
                     "High Risk — Irrigate Now"

        risk_color = "#66BB6A" if risk_score < 35 else "#FFA726" if risk_score < 65 else "#EF5350"

        ir1, ir2 = st.columns([1.5, 1])
        with ir1:
            st.markdown(f"""
            <div class="ss-card">
              <div style="font-size:0.72rem;font-weight:700;color:#8BAD87;
                          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">
                AI Irrigation Decision Engine
              </div>
              <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;">
                <div style="font-family:'DM Serif Display',serif;font-size:3rem;
                            color:{risk_color};">{risk_score}</div>
                <div>
                  <div style="font-size:0.85rem;font-weight:600;color:{risk_color};">{risk_label}</div>
                  <div style="font-size:0.75rem;color:#8BAD87;margin-top:0.2rem;">Risk Index (0–100)</div>
                </div>
              </div>
              <div style="background:#F1F8F1;border-radius:99px;height:8px;overflow:hidden;">
                <div style="width:{risk_score}%;height:100%;background:linear-gradient(90deg,#66BB6A,{risk_color});
                            border-radius:99px;transition:width 1s ease;"></div>
              </div>
              <div style="font-size:0.78rem;color:#4A6741;margin-top:0.8rem;line-height:1.5;">
                Analysis based on current humidity ({humidity}%), precipitation ({precip}mm),
                and ambient temperature ({temp}°C). {
                    'Rainfall detected — irrigation not recommended.' if precip > 0 else
                    'Conditions favor supplemental irrigation.' if risk_score > 60 else
                    'Soil moisture appears adequate for now.'
                }
              </div>
            </div>
            """, unsafe_allow_html=True)

        with ir2:
            st.markdown("### 3-Day Forecast")

            days = ["Today", "Tomorrow", "Day 3"]
            precip_sums = daily.get("precipitation_sum", [0, 0, 0])
            temp_max    = daily.get("temperature_2m_max", [28, 27, 26])
            temp_min    = daily.get("temperature_2m_min", [20, 19, 18])

            def forecast_summary(tmin, tmax, p):
                if isinstance(tmin, (int, float)) and isinstance(tmax, (int, float)):
                    if tmax >= 30:
                        temp_desc = "Hot"
                    elif tmax >= 25:
                        temp_desc = "Warm"
                    elif tmax >= 18:
                        temp_desc = "Mild"
                    else:
                        temp_desc = "Cool"
                else:
                    temp_desc = "Moderate"

                if isinstance(p, (int, float)):
                    if p >= 10:
                        rain_desc = "Heavy rain"
                    elif p >= 3:
                        rain_desc = "Showers"
                    elif p > 0:
                        rain_desc = "Light rain"
                    else:
                        rain_desc = "Dry skies"
                else:
                    rain_desc = "Stable"

                return f"{temp_desc}, {rain_desc.lower()}"

            for i, day in enumerate(days):
                p = precip_sums[i] if i < len(precip_sums) else 0
                tmax = temp_max[i] if i < len(temp_max) else None
                tmin = temp_min[i] if i < len(temp_min) else None
                icon = "🌧️" if p > 3 else "🌦️" if p > 0 else "☀️"
                summary = forecast_summary(tmin, tmax, p)
                st.markdown(f"**{icon} {day}**  \n{summary}")
                if i < len(days) - 1:
                    st.markdown("---")


# ─────────────────────────────────────────────
# VIEW D: HISTORICAL INSIGHTS & TELEMETRY LOGS
# ─────────────────────────────────────────────
def view_history():
    st.markdown('<div class="page-header">📊 Historical Insights & Telemetry Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Full activity log matrix with analytics and data portability</div>', unsafe_allow_html=True)

    df = fetch_logs()

    if df.empty:
        st.info("No activity logs found. Perform leaf scans or relay actuations to populate this view.")
        return

    # ── Summary metrics ──
    st.markdown('<div class="section-label">📈 Log Summary Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    total       = len(df)
    detections  = int((df["activity_type"] == "Detection").sum())
    actuations  = int((df["activity_type"] == "Actuation").sum())
    health_rate = round(int((df["metric_label"] == "Healthy").sum()) / max(detections, 1) * 100, 1)

    for col, icon, label, val, color in [
        (s1, "📋", "Total Log Entries",  str(total),       "#66BB6A"),
        (s2, "🔍", "Detection Events",   str(detections),  "#42A5F5"),
        (s3, "⚡", "Actuation Events",   str(actuations),  "#AB47BC"),
        (s4, "💚", "Plant Health Rate",  f"{health_rate}%","#FFA726"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{color};">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value" style="color:{color};font-size:2rem;">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Filters ──
    st.markdown('<div class="section-label">� Find journal entries</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        type_filter = st.selectbox(
            "Activity Type",
            ["All", "Detection", "Actuation"],
            key="log_type_filter"
        )
    with fc2:
        label_filter = st.selectbox(
            "Metric Label",
            ["All", "Healthy", "Sick", "Watered", "Fertilized"],
            key="log_label_filter"
        )
    with fc3:
        n_rows = st.slider("Rows to display", 5, 200, 50, key="log_n_rows")

    # Apply filters
    filtered = df.copy()
    if type_filter != "All":
        filtered = filtered[filtered["activity_type"] == type_filter]
    if label_filter != "All":
        filtered = filtered[filtered["metric_label"] == label_filter]
    filtered = filtered.head(n_rows)

    # Format display
    display_df = filtered[["timestamp", "activity_type", "event_description", "metric_label"]].copy()
    display_df.columns = ["Timestamp", "Activity Type", "Event Description", "Metric Label"]

    st.markdown('<div class="section-label">📋 Farm journal</div>', unsafe_allow_html=True)

    # Color-coded label column
    def highlight_row(row):
        color_map = {
            "Healthy":    "background-color:#E8F5E9;color:#2E7D32;",
            "Sick":       "background-color:#FFEBEE;color:#C62828;",
            "Watered":    "background-color:#E3F2FD;color:#1565C0;",
            "Fertilized": "background-color:#F3E5F5;color:#6A1B9A;",
        }
        styles = [""] * len(row)
        label_col_idx = list(row.index).index("Metric Label")
        lbl = row["Metric Label"]
        if lbl in color_map:
            styles[label_col_idx] = color_map[lbl]
        return styles

    styled = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True, height=400, hide_index=True)

    st.markdown(f"""
    <div style="font-size:0.75rem;color:#8BAD87;margin-top:0.3rem;">
      Showing {len(filtered)} of {len(df)} total records
      {f'(filtered by {type_filter} / {label_filter})' if type_filter != 'All' or label_filter != 'All' else ''}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Export ──
    st.markdown('<div class="section-label">📤 Save logs</div>', unsafe_allow_html=True)

    exp1, exp2 = st.columns([1.5, 2])
    with exp1:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        fname = f"sproutsmart_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.download_button(
            label="⬇️  Export Logs Matrix to CSV File",
            data=csv_bytes,
            file_name=fname,
            mime="text/csv",
            use_container_width=True
        )
    with exp2:
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:0.8rem 1.2rem;
                    box-shadow:0 2px 8px rgba(56,142,60,0.08);
                    border-left:3px solid #81C784;font-size:0.8rem;color:#4A6741;">
          📁 Download your farm log with all {len(df)} entries.
          The exported CSV includes Timestamp, Activity Type, Event Description, Metric Label, and Extra Value.
        </div>
        """, unsafe_allow_html=True)

    # ── Trend charts ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📉 Growth snapshots</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        label_counts = df["metric_label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        if not label_counts.empty:
            st.bar_chart(
                label_counts.set_index("Label"),
                color="#81C784",
                height=220
            )
            st.caption("Metric label distribution across all log entries")

    with chart2:
        # Activity over time (group by date)
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
        daily_counts = df_copy.groupby("date").size().reset_index(name="Events")
        if not daily_counts.empty:
            st.line_chart(
                daily_counts.set_index("date"),
                color="#66BB6A",
                height=220
            )
            st.caption("Daily event volume over time")


# ─────────────────────────────────────────────
# MAIN ROUTING LOGIC
# ─────────────────────────────────────────────
def main():
    # Initialize DB (This creates sproutsmart.db if it doesn't exist)
    init_db()
    
    if not st.session_state.get("app_loaded", False):
        render_splash_screen()
        return
    try:
        view = render_sidebar()
        if view == "🏡 Home Garden":
            view_home()
        elif view == "📸 Quick Scan":
            view_hardware()
        elif view == "🌤️ Weather Check":
            view_weather()
        elif view == "📚 Farm History":
            view_history()
    except Exception as e:
        st.error(f"System Error: {e}")
        if st.button("Clear Cache & Restart"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()