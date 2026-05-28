import serial
import time
import os
import sys
import datetime
import re

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
MOISTURE_FILE = 'moisture_data.txt'
COMMAND_FILE = 'command.txt'
FERT_TASK_FILE = 'fert_task.txt'
LOG_FILE = 'logs.txt'

def log_event(msg):
    """Timestamped logging for both console and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    print(log_entry)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry + "\n")
    except:
        pass

def send_command(command):
    """Writes a command to command.txt for the bridge to process. 
    Maintains compatibility with vision.py and pages/2_Plant_Health.py.
    """
    try:
        with open(COMMAND_FILE, 'w') as f:
            f.write(command)
        return True
    except Exception as e:
        print(f"Failed to queue command: {e}")
        return False

def main():
    log_event(f"Starting Smart Agri-Bridge on {SERIAL_PORT}...")

    while True:
        ser = None
        try:
            # 1. Attempt Serial Connection
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2) # Arduino Reset Delay
            log_event(f"Connected successfully to {SERIAL_PORT}")

            while True:
                # 2. Read Inbound Telemetry (Arduino -> PC)
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            # Extract moisture value
                            match = re.search(r'Moisture.*?:?\s*(\d+)', line)
                            if not match:
                                if "Moisture" in line:
                                    match = re.search(r'(\d+)', line)
                            
                            if match:
                                val = match.group(1)
                                with open(MOISTURE_FILE, 'w') as f:
                                    f.write(val)
                    except Exception as e:
                        log_event(f"Parsing error: {e}")

                # 3. Check for Outbound Commands (PC -> Arduino)
                # Check command.txt (Vision/Manual)
                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            cmd = f.read().strip()
                        if cmd:
                            if cmd == 'SPRAY': 
                                ser.write(b'S')
                                log_event(">>> DISPATCHED COMMAND: SPRAY [S]")
                            elif cmd == 'WATER': 
                                ser.write(b'W')
                                log_event(">>> DISPATCHED COMMAND: WATER [W]")
                            elif cmd == 'FERTILIZER' or cmd == 'F':
                                ser.write(b'F')
                                log_event(">>> DISPATCHED COMMAND: FERTILIZER [F]")
                        os.remove(COMMAND_FILE)
                    except:
                        pass

                # Check fert_task.txt (Specific for Fertilizer Button)
                if os.path.exists(FERT_TASK_FILE):
                    try:
                        ser.write(b'F')
                        log_event(">>> DISPATCHED COMMAND: FERTILIZER [F]")
                        os.remove(FERT_TASK_FILE)
                    except:
                        pass

                time.sleep(0.1) # CPU Load Balancing

        except (serial.SerialException, PermissionError) as e:
            log_event(f"Hardware Error: {e}. Retrying in 5s...")
            if ser: ser.close()
            time.sleep(5)
        
        except KeyboardInterrupt:
            log_event("Bridge stopped by user.")
            if ser: ser.close()
            sys.exit(0)
        
        except Exception as e:
            log_event(f"Unexpected Critical Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
