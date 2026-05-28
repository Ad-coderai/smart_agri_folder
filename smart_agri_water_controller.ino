/*
  Smart Garden Controller (Arduino)
  
  - Reads Soil Moisture on Pin A0
  - Prints calibrated data: "Moisture: XX%"
  - Listens for 'F' to trigger a fertilizer cycle
*/

const int sensorPin = A0;
const int relayPin = 7; // Relay for Fertilizer/Water pump

// Calibration values (Adjust these based on your specific sensor)
const int airValue = 600;   // Sensor value in dry air
const int waterValue = 250; // Sensor value in water

void setup() {
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH); // Assuming Active-Low Relay (HIGH = OFF)
  
  Serial.begin(9600);
  Serial.println(">>> Smart Garden Arduino Ready.");
}

void loop() {
  // 1. Read and Calibrate Sensor
  int rawValue = analogRead(sensorPin);
  
  // Map raw value to 0-100% (Constrain to avoid results outside bounds)
  int moisturePct = map(rawValue, airValue, waterValue, 0, 100);
  moisturePct = constrain(moisturePct, 0, 100);
  
  // 2. Report Telemetry
  Serial.print("Moisture: ");
  Serial.print(moisturePct);
  Serial.println("%");
  
  // 3. Listen for Commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'F') {
      Serial.println(">>> Command Received: [F] Triggering Fertilizer...");
      digitalWrite(relayPin, LOW); // ON
      delay(3000);                 // Run for 3 seconds
      digitalWrite(relayPin, HIGH); // OFF
    }
  }
  
  delay(1000); // Wait 1 second between reads
}
