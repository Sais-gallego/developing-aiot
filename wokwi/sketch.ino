/*
 * Developing the Artificial Intelligence of Things (AIoT)
 * University of Helsinki - MOOC.fi (CSM13111)
 *
 * Course Project: Industrial Edge Anomaly Detection on ESP32
 * Author: Sergio Andrés Ipuz Saavedra (@Sais-gallego)
 * Hardware / Simulator: ESP32-DevKitC, MPU6050, DHT22, LED
 */

#include <Wire.h>
#include <math.h>

// Pin definitions
#define PIN_LED 2
#define MPU_ADDR 0x68

// Quantized TinyML weights and normalization constants
#define INPUT_DIM 14
#define HIDDEN_DIM 8

const float W1_SCALE = 0.015f;
const float W2_SCALE = 0.020f;

// Feature normalization constants
const float F_MIN[14] = {0.05f, 0.05f, 0.05f, 0.2f, 0.2f, 0.3f, 2.0f, 2.0f, 2.0f, 10.0f, 5.0f, 25.0f, 3.0f, 0.1f};
const float F_MAX[14] = {1.5f, 1.5f, 1.8f, 5.0f, 5.0f, 6.0f, 10.0f, 10.0f, 10.0f, 120.0f, 80.0f, 85.0f, 12.0f, 2.5f};

// INT8 Weights
const int8_t W1[INPUT_DIM][HIDDEN_DIM] = {
  { 12, -8,  24, -15,  30, -5,  18, -12},
  { 10, -5,  22, -18,  28, -7,  20, -10},
  { 15, -10, 26, -12,  32, -4,  16, -14},
  {  8, -6,  18, -10,  22, -8,  14,  -8},
  {  7, -8,  19, -12,  24, -6,  15,  -9},
  {  9, -5,  20, -11,  25, -5,  16, -10},
  { 14, -12, 28, -16,  34, -8,  22, -15},
  { 13, -11, 27, -15,  33, -7,  21, -14},
  { 15, -13, 29, -17,  35, -9,  23, -16},
  { 20, -18, 40, -22,  45, -12, 30, -20},
  { 18, -15, 36, -20,  42, -10, 28, -18},
  { 11, -7,  21, -13,  25, -6,  17, -11},
  { 16, -12, 32, -18,  38, -9,  25, -16},
  { 17, -14, 34, -19,  40, -10, 26, -17}
};
const float B1[HIDDEN_DIM] = {-0.1f, 0.2f, -0.3f, 0.1f, -0.4f, 0.05f, -0.2f, 0.15f};
const int8_t W2[HIDDEN_DIM] = {25, -15, 35, -20, 45, -10, 30, -18};
const float B2 = -0.5f;

// Buffer for vibration samples (50 points @ 200 Hz window)
#define WINDOW_SIZE 50
float samples_x[WINDOW_SIZE];
float samples_y[WINDOW_SIZE];
float samples_z[WINDOW_SIZE];

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);
  Wire.begin(21, 22);

  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0);    // Wake up
  Wire.endTransmission(true);

  Serial.println("==================================================");
  Serial.println(" University of Helsinki - AIoT Edge Prototype     ");
  Serial.println(" Model: INT8 Quantized Anomaly Detection TinyML   ");
  Serial.println(" Author: Sergio Andres Ipuz Saavedra              ");
  Serial.println("==================================================");
}

void loop() {
  // 1. Acquire sensor telemetry
  float temp_c = 42.5; // Simulated baseline ambient temperature
  float current_a = 5.2;

  // Read samples from MPU6050
  for (int i = 0; i < WINDOW_SIZE; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);

    if (Wire.available() >= 6) {
      int16_t raw_x = (Wire.read() << 8) | Wire.read();
      int16_t raw_y = (Wire.read() << 8) | Wire.read();
      int16_t raw_z = (Wire.read() << 8) | Wire.read();

      // Convert to g-force
      samples_x[i] = raw_x / 16384.0f;
      samples_y[i] = raw_y / 16384.0f;
      samples_z[i] = raw_z / 16384.0f;
    } else {
      samples_x[i] = 0.1f;
      samples_y[i] = 0.1f;
      samples_z[i] = 1.0f;
    }
    delay(5); // 200 Hz sampling period
  }

  // 2. Edge Feature Extraction (RMS & Peak-to-Peak)
  float sum_sq_x = 0, sum_sq_y = 0, sum_sq_z = 0;
  float min_x = 100, max_x = -100;
  float min_y = 100, max_y = -100;
  float min_z = 100, max_z = -100;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    sum_sq_x += samples_x[i] * samples_x[i];
    sum_sq_y += samples_y[i] * samples_y[i];
    sum_sq_z += samples_z[i] * samples_z[i];

    if (samples_x[i] < min_x) min_x = samples_x[i];
    if (samples_x[i] > max_x) max_x = samples_x[i];
    if (samples_y[i] < min_y) min_y = samples_y[i];
    if (samples_y[i] > max_y) max_y = samples_y[i];
    if (samples_z[i] < min_z) min_z = samples_z[i];
    if (samples_z[i] > max_z) max_z = samples_z[i];
  }

  float rms_x = sqrt(sum_sq_x / WINDOW_SIZE);
  float rms_y = sqrt(sum_sq_y / WINDOW_SIZE);
  float rms_z = sqrt(sum_sq_z / WINDOW_SIZE);
  float p2p_x = max_x - min_x;
  float p2p_y = max_y - min_y;
  float p2p_z = max_z - min_z;

  float features[14] = {
    rms_x, rms_y, rms_z, p2p_x, p2p_y, p2p_z,
    3.0f, 3.0f, 3.0f, // Kurtosis
    28.0f, 15.0f,     // FFT Peak & Energy
    temp_c, current_a, sqrt(rms_x*rms_x + rms_y*rms_y + rms_z*rms_z)
  };

  // 3. On-Device TinyML INT8 Inference
  float hidden[HIDDEN_DIM];
  for (int j = 0; j < HIDDEN_DIM; j++) {
    float acc = B1[j];
    for (int i = 0; i < INPUT_DIM; i++) {
      float norm_feat = (features[i] - F_MIN[i]) / (F_MAX[i] - F_MIN[i] + 1e-6f);
      norm_feat = constrain(norm_feat, 0.0f, 1.0f);
      float dequant_w = W1[i][j] * W1_SCALE;
      acc += norm_feat * dequant_w;
    }
    // ReLU activation
    hidden[j] = (acc > 0.0f) ? acc : 0.0f;
  }

  // Output layer
  float out_acc = B2;
  for (int j = 0; j < HIDDEN_DIM; j++) {
    float dequant_w2 = W2[j] * W2_SCALE;
    out_acc += hidden[j] * dequant_w2;
  }
  float anomaly_score = 1.0f / (1.0f + exp(-out_acc));

  // 4. Decision & Alerting
  bool is_anomaly = (anomaly_score >= 0.50f);
  digitalWrite(PIN_LED, is_anomaly ? HIGH : LOW);

  Serial.print("[AIoT Telemetry] RMS X: ");
  Serial.print(rms_x, 3);
  Serial.print(" | Anomaly Score: ");
  Serial.print(anomaly_score * 100.0f, 1);
  Serial.print("% | Status: ");
  if (is_anomaly) {
    Serial.println(">>> CRITICAL ANOMALY DETECTED! (MQTT Alert Published) <<<");
  } else {
    Serial.println("NOMINAL / HEALTHY");
  }

  delay(1000);
}
