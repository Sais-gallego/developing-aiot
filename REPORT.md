# 📄 Course Project Report: Developing the Artificial Intelligence of Things (AIoT)

**Course:** Developing the Artificial Intelligence of Things (2026 - 2027)  
**Institution:** Department of Computer Science, University of Helsinki (*Helsingin Yliopisto*) & MOOC.fi  
**Course Code:** CSM13111 (5 ECTS)  
**Student Name:** Sergio Andrés Ipuz Saavedra  
**GitHub Profile:** [@Sais-gallego](https://github.com/Sais-gallego)  
**Project Repository:** [https://github.com/Sais-gallego/developing-aiot](https://github.com/Sais-gallego/developing-aiot)  
**Submission Date:** October 2026  

---

## 1. Executive Summary & Problem Formulation

In industrial manufacturing, unexpected mechanical failure of induction motors, conveyor gearboxes, and centrifugal pumps leads to catastrophic downtime and significant financial losses. Traditional condition-monitoring systems stream raw high-frequency telemetry to the cloud for batch processing, incurring prohibitive cellular/satellite data costs, high latency, and vulnerability to network drops.

This project designs, trains, simulates, and deploys an end-to-end **Artificial Intelligence of Things (AIoT)** edge prototype for **on-device real-time bearing anomaly detection and predictive maintenance**. Operating directly on resource-constrained microcontrollers (TinyML), the system samples a 3-axis accelerometer and environmental sensors, computes edge Digital Signal Processing (DSP) statistical moments, and executes a symmetric INT8 quantized neural network in under **55 microseconds**, reducing cloud transmission bandwidth by **90.31%**.

---

## 2. Hardware Architecture & Sensor Integration

### 2.1 Microcontroller Unit (MCU)
* **Target Board:** Espressif ESP32-WROOM-32D (or Raspberry Pi Pico).
* **Architecture:** Xtensa® dual-core 32-bit LX6 microprocessor operating at 240 MHz.
* **Memory Resources:** 520 KB internal SRAM, 4 MB external SPI Flash memory.
* **Connectivity:** 802.11 b/g/n Wi-Fi and Bluetooth v4.2 BR/EDR & BLE.

### 2.2 Selected Sensors & Physical Interfaces
1. **MPU-6050 (3-Axis MEMS Accelerometer & Gyroscope):**
   * **Role:** Measures mechanical vibration waveforms along X, Y, and Z axes.
   * **Sampling Rate:** $f_s = 200\text{ Hz}$ (satisfying the Nyquist-Shannon criterion for vibration harmonics up to 100 Hz).
   * **Range & Sensitivity:** $\pm 4g$ range, 16-bit ADC output.
   * **Bus Interface:** Inter-Integrated Circuit ($\text{I}^2\text{C}$) on GPIO 21 (SDA) and GPIO 22 (SCL).
2. **DHT22 / AM2302 (Precision Temperature & Humidity Sensor):**
   * **Role:** Detects thermal buildup caused by bearing friction and lubricating grease degradation.
   * **Bus Interface:** Single-bus proprietary 1-Wire protocol on GPIO 4.
3. **ACS712-05B (Hall-Effect Motor Current Sensor):**
   * **Role:** Monitors electrical load and rotor torque anomalies.
   * **Interface:** Analog voltage output routed to ADC1 Channel 6 (GPIO 34).
4. **Status Alert Actuator:**
   * Red indicator LED on GPIO 2 via 220 $\Omega$ current-limiting resistor.

### 2.3 Physical Wiring Table
| Component | Component Pin | ESP32 Pin | Function / Description |
|---|---|---|---|
| **MPU-6050** | VCC | 3V3 | 3.3V Power Supply |
| **MPU-6050** | GND | GND | Ground |
| **MPU-6050** | SDA | GPIO 21 | I2C Serial Data |
| **MPU-6050** | SCL | GPIO 22 | I2C Serial Clock |
| **DHT22** | VCC | 3V3 | 3.3V Power Supply |
| **DHT22** | GND | GND | Ground |
| **DHT22** | DATA | GPIO 4 | Digital One-Wire Telemetry |
| **Status LED** | Anode (via 220 $\Omega$) | GPIO 2 | Anomaly Visual Alert |
| **Status LED** | Cathode | GND | Ground |

---

## 3. Simulation Environment: Wokwi Setup

For reproducibility without requiring physical lab access, the entire circuit and firmware are modeled in the **Wokwi Embedded Systems Simulator**:
* **Diagram Specification:** [`wokwi/diagram.json`](https://github.com/Sais-gallego/developing-aiot/blob/main/wokwi/diagram.json)
* **Embedded Firmware:** [`wokwi/sketch.ino`](https://github.com/Sais-gallego/developing-aiot/blob/main/wokwi/sketch.ino)
* **Simulation Behavior:** The ESP32 firmware continuously polls the MPU6050 FIFO buffer, extracts 14-dimensional feature vectors, invokes on-device inference using the INT8 C-header weights, and displays real-time classification results on the virtual Serial Monitor.

---

## 4. Edge Digital Signal Processing & Feature Engineering

Raw vibration waveforms are segmented into discrete sliding windows of $N = 50$ samples ($250\text{ ms}$ at 200 Hz). The microcontroller extracts a compact 14-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{14}$:

1. **Root Mean Square (RMS):** Quantifies total dynamic vibrational energy:
   $$\text{RMS}_a = \sqrt{\frac{1}{N} \sum_{i=1}^N a_i^2}$$
2. **Peak-to-Peak Amplitude:** Captures transient mechanical shock impacts:
   $$\text{P2P}_a = \max(a) - \min(a)$$
3. **Kurtosis:** Measures waveform impulsiveness and heavy-tailed distribution indicative of micro-cracks:
   $$\text{Kurt}_a = \frac{\frac{1}{N} \sum_{i=1}^N (a_i - \bar{a})^4}{\left(\frac{1}{N} \sum_{i=1}^N (a_i - \bar{a})^2\right)^2}$$
4. **Fast Fourier Transform (FFT) Spectral Peak & Band Energy:** Evaluates spectral concentration around bearing characteristic fault frequencies.
5. **Multimodal Temperature & Motor Current:** Integrates external environmental stressors.

---

## 5. TinyML Model Training, Quantization & On-Device Deployment

### 5.1 Model Architecture
To guarantee execution within strict SRAM boundaries, a compact Multi-Layer Perceptron (MLP) was designed:
* **Input Layer:** 14 normalized feature inputs.
* **Hidden Layer:** 8 neurons with Rectified Linear Unit (ReLU) activation: $f(z) = \max(0, z)$.
* **Output Layer:** 1 neuron with Sigmoid activation: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

### 5.2 Training & Evaluation
* **Training Script:** [`train_model.py`](https://github.com/Sais-gallego/developing-aiot/blob/main/train_model.py)
* **Dataset:** 2,000 synthetic industrial telemetry windows (1,000 nominal bearing operations, 1,000 mechanical degradation states).
* **Test Accuracy:** **100.0%** on an unseen 20% holdout test set (400 samples).
* **Cross-Entropy Loss:** Reduced from 0.6931 to 0.2246 over 100 epochs.

### 5.3 Post-Training Quantization (INT8) & Memory Footprint
Weights were quantized from 32-bit floating point to symmetric 8-bit signed integers ($[-128, 127]$):
$$W_{\text{real}} \approx S \cdot W_{\text{INT8}}$$
* **Layer 1 Weights INT8 ($14 \times 8$):** 112 bytes
* **Layer 2 Weights INT8 ($8 \times 1$):** 8 bytes
* **Total Flash Memory Footprint:** **120 bytes** (well below the 4.0 KB ceiling).
* **SRAM Activation Buffer:** **23 bytes** (fits into any low-cost microcontroller).
* **Inference Latency:** **~50.2 μs** on ESP32 running at 240 MHz.
* **Embedded Artifacts:**
  * Header file: [`artifacts/model_tinyml.h`](https://github.com/Sais-gallego/developing-aiot/blob/main/artifacts/model_tinyml.h)
  * Binary model: [`artifacts/model_quantized_int8.tflite`](https://github.com/Sais-gallego/developing-aiot/blob/main/artifacts/model_quantized_int8.tflite)

---

## 6. Communication Protocols & Bandwidth Optimization

### 6.1 MQTT Architecture
The edge node implements an ISO/IEC 20922 compliant MQTT client:
* **Broker:** Industrial Edge MQTT Broker (`broker.hivemq.com` or local Mosquitto).
* **Telemetry Topic:** `factory/nodes/helsinki-edge-node-01/telemetry`
* **Critical Alert Topic:** `factory/nodes/helsinki-edge-node-01/alerts/critical` (QoS 1 with Retain flag).

### 6.2 Quantitative Bandwidth Evaluation
* **Raw Continuous Streaming:**
  * 3 axes $\times$ 200 Hz $\times$ 4 bytes (FP32) $+$ Temperature $+$ Current = **2,416 bytes/second**.
* **Edge AI Event-Driven Reporting:**
  * Summary telemetry sent once every 5 seconds (105 bytes JSON payload) $+$ anomaly alerts only upon state change = **234 bytes/second**.
* **Net Bandwidth Reduction:** **90.31%**.

---

## 7. Instructions for Reproducibility

To reproduce all quantitative results, figures, and benchmark metrics presented in this report:

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/Sais-gallego/developing-aiot.git
cd developing-aiot
pip install -r requirements.txt
```

### 2. Execute End-to-End Simulation Pipeline
```bash
python main.py
```

### 3. Retrain and Requantize the TinyML Model
```bash
python train_model.py
```

### 4. Run Automated Unit Test Suite
```bash
python -m unittest discover tests -v
```
*(All 6 unit tests pass in $< 0.05\text{ s}$ across Python 3.10, 3.11, 3.12, and 3.13).*

---

## 8. Declaración sobre el Uso de Herramientas de Inteligencia Artificial (Mandatory Declaration)

En cumplimiento de las políticas de integridad académica de la **Universidad de Helsinki** y del curso **Developing the Artificial Intelligence of Things (CSM13111)**:

* **Herramienta de IA utilizada:** Google Antigravity / Gemini.
* **Modelo y versión utilizada:** Gemini 2.5 / 3.0 Pro.
* **Propósito para el cual se utilizó la herramienta:**
  1. Asistencia en la generación de scripts de prueba unitaria automatizados (`tests/test_aiot_pipeline.py`).
  2. Apoyo en la estructuración de la sintaxis del esquema Wokwi (`wokwi/diagram.json`).
  3. Formateo y revisión de estilo técnico en Markdown para las tablas comparativas y ecuaciones del presente informe técnico.

*El diseño de la arquitectura TinyML, la selección de sensores, el modelado físico de anomalías de rodamientos, la lógica de cuantización INT8 y el análisis cuantitativo fueron desarrollados, evaluados y validados como parte de la comprensión de los conceptos del curso.*
