/*
 * Universal E-Nose Sensor Data Acquisition
 * Specifically configured for your 11-sensor array:
 * - SEN0515 (ENS160) - TVOC/eCO2/AQI
 * - SEN0536 (SCD41) - CO2
 * - SEN0569 (MEMS Ethanol) - Alcohol
 * - SEN0567 (MEMS NH3) - Ammonia
 * - SEN0574 (MEMS NO2) - Nitrogen dioxide
 * - SEN0466 (Electrochemical CO) - Carbon monoxide
 * - SEN0572 (MEMS H2) - Hydrogen
 * - SEN0565 (MEMS CH4) - Methane
 * - SEN0570 (MEMS Smoke) - Smoke detection
 * - MQ-135 - Air quality
 * - MQ-3 - Alcohol/benzene
 */

#include <Wire.h>
#include <Adafruit_SCD4x.h>
#include <SparkFun_ENS160.h>

// I2C Sensors
Adafruit_SCD4x scd4x;
SparkFun_ENS160 ens160;

// Analog Sensors (A0-A8 for 9 analog sensors)
const int NUM_ANALOG_SENSORS = 9;
const int analogPins[NUM_ANALOG_SENSORS] = {A0, A1, A2, A3, A4, A5, A6, A7, A8};

// Calibration values
float baselineValues[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
bool isCalibrated = false;

unsigned long lastReadTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  // Initialize I2C
  Wire.begin();

  // Initialize SCD41
  if (!scd4x.begin()) {
    Serial.println("SCD41 not found!");
  } else {
    scd4x.startPeriodicMeasurement();
    Serial.println("SCD41 initialized");
  }

  // Initialize ENS160
  if (!ens160.begin()) {
    Serial.println("ENS160 not found!");
  } else {
    ens160.setMode(ENS160_MODE_STANDARD);
    Serial.println("ENS160 initialized");
  }

  Serial.println("Universal E-Nose System Ready");
  Serial.println("Commands: read, stream, stop, calibrate, status, sample");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "read") {
      readAndOutputSensors(false);
    }
    else if (command == "stream") {
      streamSensors();
    }
    else if (command == "stop") {
      Serial.println("Streaming stopped");
    }
    else if (command == "calibrate") {
      performCalibration();
    }
    else if (command == "status") {
      showCalibrationStatus();
    }
    else if (command == "sample") {
      performExtendedSampling();
    }
    else if (command == "raw") {
      outputRawValues();
    }
    else {
      Serial.println("Unknown command. Available: read, stream, stop, calibrate, status, sample, raw");
    }
  }
}

float readSCD41() {
  if (scd4x.getDataReady()) {
    uint16_t co2 = scd4x.readCO2();
    return min(co2 / 5000.0, 1.0); // Normalize 0-5000ppm to 0-1
  }
  return 0.0;
}

float readENS160() {
  if (ens160.checkStatus() == ENS160_STATUS_OK) {
    uint16_t tvoc = ens160.readTVOC();
    return tvoc / 65535.0; // Normalize 0-65535 to 0-1
  }
  return 0.0;
}

float readAnalogSensor(int pin) {
  int rawValue = analogRead(pin);
  return min(rawValue / 1023.0, 1.0); // Normalize 0-1023 to 0-1
}

void readAllSensors(float sensorValues[11]) {
  // Digital sensors (I2C)
  sensorValues[0] = readENS160();    // SEN0515 - TVOC
  sensorValues[1] = readSCD41();     // SEN0536 - CO2

  // Analog sensors (A0-A8)
  sensorValues[2] = readAnalogSensor(A0); // SEN0569 - Ethanol
  sensorValues[3] = readAnalogSensor(A1); // SEN0567 - NH3
  sensorValues[4] = readAnalogSensor(A2); // SEN0574 - NO2
  sensorValues[5] = readAnalogSensor(A3); // SEN0466 - CO
  sensorValues[6] = readAnalogSensor(A4); // SEN0572 - H2
  sensorValues[7] = readAnalogSensor(A5); // SEN0565 - CH4
  sensorValues[8] = readAnalogSensor(A6); // SEN0570 - Smoke
  sensorValues[9] = readAnalogSensor(A7); // MQ-135
  sensorValues[10] = readAnalogSensor(A8); // MQ-3
}

void readAndOutputSensors(bool includeTimestamp = false) {
  float sensorValues[11];
  readAllSensors(sensorValues);

  // Apply calibration if available
  if (isCalibrated) {
    for (int i = 0; i < 11; i++) {
      sensorValues[i] = sensorValues[i] - (baselineValues[i] / 1023.0);
      if (sensorValues[i] < 0.0) sensorValues[i] = 0.0;
    }
  }

  outputJSON(sensorValues, includeTimestamp);
}

void outputJSON(float values[11], bool includeTimestamp = false) {
  Serial.print("{\"sensors\":[");
  for (int i = 0; i < 11; i++) {
    Serial.print(values[i], 4);
    if (i < 10) Serial.print(",");
  }
  Serial.print("]");

  if (includeTimestamp) {
    Serial.print(",\"timestamp\":");
    Serial.print(millis());
  }
  Serial.println("}");
}

void streamSensors() {
  Serial.println("Starting continuous streaming...");
  Serial.println("Send 'stop' to halt streaming");

  while (true) {
    unsigned long currentTime = millis();
    if (currentTime - lastReadTime >= 100) { // 10 Hz
      readAndOutputSensors(true);
      lastReadTime = currentTime;
    }

    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\n');
      command.trim();
      if (command == "stop") {
        break;
      }
    }
    delay(10);
  }
  Serial.println("Streaming stopped");
}

void performCalibration() {
  const int CALIBRATION_SAMPLES = 200;
  const int CALIBRATION_DELAY = 50;

  Serial.println("Starting baseline calibration...");
  Serial.println("Ensure clean air environment!");
  Serial.println("Collecting " + String(CALIBRATION_SAMPLES) + " samples...");

  float sumValues[11] = {0};

  for (int sample = 0; sample < CALIBRATION_SAMPLES; sample++) {
    float sensorReadings[11];
    readAllSensors(sensorReadings);

    for (int i = 0; i < 11; i++) {
      sumValues[i] += sensorReadings[i] * 1023.0; // Convert back to raw scale
    }
    delay(CALIBRATION_DELAY);

    if (sample % 20 == 0) {
      Serial.print(".");
    }
  }

  for (int i = 0; i < 11; i++) {
    baselineValues[i] = sumValues[i] / CALIBRATION_SAMPLES;
  }

  isCalibrated = true;
  Serial.println();
  Serial.println("Calibration completed!");
  showCalibrationStatus();
}

void performExtendedSampling() {
  const int MAX_DURATION = 10000; // 10 seconds
  const int SAMPLE_INTERVAL = 100;

  Serial.println("Starting extended sampling mode...");
  Serial.println("Collecting data for up to 10 seconds...");

  unsigned long startTime = millis();
  int sampleCount = 0;

  while (millis() - startTime < MAX_DURATION) {
    if ((millis() - lastReadTime) >= SAMPLE_INTERVAL) {
      readAndOutputSensors(true);
      lastReadTime = millis();
      sampleCount++;
    }
    delay(10);
  }

  Serial.println("Extended sampling completed!");
  Serial.println("Samples collected: " + String(sampleCount));
}

void showCalibrationStatus() {
  if (isCalibrated) {
    Serial.println("Calibration Status: CALIBRATED");
    Serial.print("Baseline values (normalized): [");
    for (int i = 0; i < 11; i++) {
      Serial.print(baselineValues[i] / 1023.0, 4);
      if (i < 10) Serial.print(", ");
    }
    Serial.println("]");
  } else {
    Serial.println("Calibration Status: NOT CALIBRATED");
    Serial.println("Run 'calibrate' command to perform baseline calibration");
  }
}

void outputRawValues() {
  float sensorValues[11];
  readAllSensors(sensorValues);

  Serial.print("Normalized values (0-1.0): [");
  for (int i = 0; i < 11; i++) {
    Serial.print(sensorValues[i], 4);
    if (i < 10) Serial.print(", ");
  }
  Serial.println("]");
}
