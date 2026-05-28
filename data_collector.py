"""data_collector.py

Robust Python bridge between Arduino (Serial) and Streamlit.

- Continuously reads moisture data from the Arduino on COM3 at 9600 baud.
- Parses the "Moisture%" value from lines like:
    Raw:512 | Moisture%:67 | Relay:ON 67
  (and also falls back to the last integer token if needed)
- Stores the *latest* reading to a local JSON file.
- Handles serial connection errors gracefully (Arduino unplugged, COM busy, etc.)

Usage:
  1) Terminal 1:  python data_collector.py
  2) Terminal 2:  streamlit run main.py

If Arduino is unplugged, Streamlit will show connection status without crashing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple


try:
    import serial  # type: ignore
    from serial import SerialException  # type: ignore
except Exception:  # pragma: no cover
    serial = None
    SerialException = Exception  # type: ignore


DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT_S = 1.0

# Where to write the latest value for the Streamlit app to read.
DEFAULT_LATEST_JSON_PATH = "latest_moisture.json"


# Match Moisture%:<number>
MOISTURE_PCT_RE = re.compile(r"Moisture%\s*[:=]\s*(-?\d+)")


@dataclass
class LatestPayload:
    moisture_pct: Optional[int]
    relay_on: Optional[bool]
    raw: Optional[int]
    timestamp_epoch: float
    timestamp_iso: str
    connected: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "moisture_pct": self.moisture_pct,
            "relay_on": self.relay_on,
            "raw": self.raw,
            "timestamp_epoch": self.timestamp_epoch,
            "timestamp_iso": self.timestamp_iso,
            "connected": self.connected,
            "error": self.error,
        }


def _utc_now_iso() -> Tuple[float, str]:
    ts_epoch = time.time()
    ts_iso = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()
    return ts_epoch, ts_iso


def _safe_int(token: str) -> Optional[int]:
    try:
        return int(token)
    except Exception:
        return None


def parse_arduino_line(line: str) -> Tuple[Optional[int], Optional[bool], Optional[int]]:
    """Parse an Arduino line.

    Returns (moisture_pct, relay_on, raw)
    """

    # Example line from sketch:
    # Raw:500 | Moisture%:70 | Relay:OFF 70
    moisture_pct: Optional[int] = None
    raw_val: Optional[int] = None
    relay_on: Optional[bool] = None

    m = MOISTURE_PCT_RE.search(line)
    if m:
        moisture_pct = _safe_int(m.group(1))

    # Raw:
    m_raw = re.search(r"Raw\s*[:=]\s*(-?\d+)", line)
    if m_raw:
        raw_val = _safe_int(m_raw.group(1))

    # Relay:ON or Relay:OFF
    if "Relay:" in line:
        if "Relay:ON" in line:
            relay_on = True
        elif "Relay:OFF" in line:
            relay_on = False

    # Fallback: last integer token is often the moisture % per user description.
    if moisture_pct is None:
        ints = re.findall(r"-?\d+", line)
        if ints:
            moisture_pct = _safe_int(ints[-1])

    return moisture_pct, relay_on, raw_val


class ArduinoCollector:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        latest_json_path: str = DEFAULT_LATEST_JSON_PATH,
        reconnect_backoff_s: float = 2.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.latest_json_path = latest_json_path
        self.reconnect_backoff_s = reconnect_backoff_s

        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _write_payload(self, payload: LatestPayload) -> None:
        tmp_path = self.latest_json_path + ".tmp"
        data = payload.to_dict()
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, self.latest_json_path)

    def run_forever(self) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run: pip install -r requirements.txt"
            )

        # Initialize file so Streamlit has something to read.
        ts_epoch, ts_iso = _utc_now_iso()
        self._write_payload(
            LatestPayload(
                moisture_pct=None,
                relay_on=None,
                raw=None,
                timestamp_epoch=ts_epoch,
                timestamp_iso=ts_iso,
                connected=False,
                error="collector_not_started",
            )
        )

        while not self._stop_event.is_set():
            try:
                with serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout_s,
                ) as ser:
                    # Mark connected
                    ts_epoch, ts_iso = _utc_now_iso()
                    self._write_payload(
                        LatestPayload(
                            moisture_pct=None,
                            relay_on=None,
                            raw=None,
                            timestamp_epoch=ts_epoch,
                            timestamp_iso=ts_iso,
                            connected=True,
                            error=None,
                        )
                    )

                    # Read loop
                    while not self._stop_event.is_set():
                        raw_line = ser.readline()  # bytes
                        if not raw_line:
                            continue

                        try:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            continue

                        if not line:
                            continue

                        moisture_pct, relay_on, raw_val = parse_arduino_line(line)
                        if moisture_pct is None:
                            # Ignore lines that don't include moisture.
                            continue

                        ts_epoch, ts_iso = _utc_now_iso()
                        payload = LatestPayload(
                            moisture_pct=moisture_pct,
                            relay_on=relay_on,
                            raw=raw_val,
                            timestamp_epoch=ts_epoch,
                            timestamp_iso=ts_iso,
                            connected=True,
                            error=None,
                        )
                        self._write_payload(payload)

            except SerialException as e:
                ts_epoch, ts_iso = _utc_now_iso()
                payload = LatestPayload(
                    moisture_pct=None,
                    relay_on=None,
                    raw=None,
                    timestamp_epoch=ts_epoch,
                    timestamp_iso=ts_iso,
                    connected=False,
                    error=str(e),
                )
                self._write_payload(payload)
                time.sleep(self.reconnect_backoff_s)
            except OSError as e:
                ts_epoch, ts_iso = _utc_now_iso()
                payload = LatestPayload(
                    moisture_pct=None,
                    relay_on=None,
                    raw=None,
                    timestamp_epoch=ts_epoch,
                    timestamp_iso=ts_iso,
                    connected=False,
                    error=str(e),
                )
                self._write_payload(payload)
                time.sleep(self.reconnect_backoff_s)
            except Exception as e:
                # Prevent hard crashes; keep retrying.
                ts_epoch, ts_iso = _utc_now_iso()
                payload = LatestPayload(
                    moisture_pct=None,
                    relay_on=None,
                    raw=None,
                    timestamp_epoch=ts_epoch,
                    timestamp_iso=ts_iso,
                    connected=False,
                    error=f"unexpected: {e}",
                )
                try:
                    self._write_payload(payload)
                except Exception:
                    pass
                time.sleep(self.reconnect_backoff_s)


def get_latest_moisture_value(latest_json_path: str = DEFAULT_LATEST_JSON_PATH) -> LatestPayload:
    """Read the latest moisture payload from JSON.

    Streamlit calls this frequently (fast, no serial I/O).
    """

    if not os.path.exists(latest_json_path):
        ts_epoch, ts_iso = _utc_now_iso()
        return LatestPayload(
            moisture_pct=None,
            relay_on=None,
            raw=None,
            timestamp_epoch=ts_epoch,
            timestamp_iso=ts_iso,
            connected=False,
            error="latest_json_missing",
        )

    try:
        with open(latest_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Defensive parsing
        return LatestPayload(
            moisture_pct=data.get("moisture_pct"),
            relay_on=data.get("relay_on"),
            raw=data.get("raw"),
            timestamp_epoch=float(data.get("timestamp_epoch", 0.0)),
            timestamp_iso=str(data.get("timestamp_iso", "")),
            connected=bool(data.get("connected", False)),
            error=data.get("error"),
        )
    except Exception as e:
        ts_epoch, ts_iso = _utc_now_iso()
        return LatestPayload(
            moisture_pct=None,
            relay_on=None,
            raw=None,
            timestamp_epoch=ts_epoch,
            timestamp_iso=ts_iso,
            connected=False,
            error=f"read_error: {e}",
        )


if __name__ == "__main__":
    # Optional CLI args:
    #   python data_collector.py COM3 9600 latest_moisture.json
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BAUD
    out_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LATEST_JSON_PATH

    collector = ArduinoCollector(port=port, baudrate=baud, latest_json_path=out_path)
    print(f"[data_collector] Starting serial collector on {port} @ {baud} baud...")
    collector.run_forever()

