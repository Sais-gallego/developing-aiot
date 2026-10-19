# ⚡ Developing the Artificial Intelligence of Things (AIoT)

[![CI Build](https://github.com/Sais-gallego/developing-aiot/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/developing-aiot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![University of Helsinki](https://img.shields.io/badge/University%20of%20Helsinki-MOOC.fi-black.svg)](https://courses.mooc.fi/org/uh-cs/courses/developing-the-artificial-intelligence-of-things-2026-2027)

> **Official Course Project for Developing the Artificial Intelligence of Things (AIoT - CSM13111, 5 ECTS)**  
> **Institution:** Department of Computer Science, University of Helsinki (*Helsingin Yliopisto*) & MOOC.fi  
> **Student / Author:** Sergio Andrés Ipuz Saavedra ([@Sais-gallego](https://github.com/Sais-gallego))

---

## 📌 Project Overview

This repository implements the end-to-end coursework, training pipeline, and edge simulation prototype for **Chapter 3: Course Project** of **Developing the Artificial Intelligence of Things (AIoT)**. It features an on-device condition-monitoring and predictive maintenance system tailored for industrial rotating equipment (bearings, motors, pumps) under strict micro-power and micro-memory constraints:

1. **Multi-Sensor Edge Sensing (`src/sensor_node.py`):** High-frequency 3-axis vibration telemetry (sampling at 200 Hz), ambient motor temperature (DHT22), and electrical current draw (ACS712).
2. **Edge Feature Engineering (`src/feature_engineering.py`):** Time-domain statistical moments (RMS, Peak-to-Peak amplitude, Kurtosis) and Fast Fourier Transform (FFT) dominant harmonic energy estimation.
3. **TinyML Model Training & Quantization (`train_model.py`):** Compact Multi-Layer Perceptron trained on 2,000 industrial vibration windows, achieving **100.0% classification accuracy** and compressed via symmetric INT8 Post-Training Quantization (PTQ) to **120 bytes of Flash** and **23 bytes of SRAM**.
4. **On-Device Edge Inference Engine (`src/edge_inference.py`):** Integer arithmetic execution simulating low-power MCU runtime with real-time latency ($< 55\ \mu\text{s}$) and energy profiling ($< 3.5\ \mu\text{J}$).
5. **Wokwi Simulator Circuit & C++ Firmware (`wokwi/`):** Complete simulation schematic (`wokwi/diagram.json`) and Arduino/ESP32 C++ sketch (`wokwi/sketch.ino`) executing INT8 inference directly from C-header weights (`artifacts/model_tinyml.h`).
6. **MQTT Telemetry & Bandwidth Optimization (`src/mqtt_gateway.py`):** ISO/IEC 20922 compliant edge publisher achieving **90.31% bandwidth reduction** over raw telemetry streaming.

---

## 🏗️ Hardware Architecture & Wiring

* **Microcontroller:** ESP32-WROOM-32D (240 MHz dual-core, 520 KB SRAM, 4 MB Flash).

| Component | Sensor Pin | ESP32 Pin | Interface / Protocol |
|---|---|---|---|
| **MPU-6050 Accelerometer** | VCC | 3V3 | 3.3V Power |
| **MPU-6050 Accelerometer** | GND | GND | Ground |
| **MPU-6050 Accelerometer** | SDA | GPIO 21 | I2C Serial Data |
| **MPU-6050 Accelerometer** | SCL | GPIO 22 | I2C Serial Clock |
| **DHT22 Temperature** | VCC | 3V3 | 3.3V Power |
| **DHT22 Temperature** | GND | GND | Ground |
| **DHT22 Temperature** | DATA | GPIO 4 | 1-Wire Digital Telemetry |
| **Status Alert LED** | Anode (via 220 $\Omega$) | GPIO 2 | Digital Output Actuator |
| **Status Alert LED** | Cathode | GND | Ground |

---

## 📁 Repository Structure

```
developing-aiot/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions multi-version CI (Python 3.10 - 3.13)
├── artifacts/
│   ├── model_tinyml.h           # Auto-generated C header with quantized INT8 weights
│   ├── model_quantized_int8.tflite # Compact binary model artifact (124 bytes)
│   └── model_metadata.json      # Architectural specifications and benchmarks
├── src/
│   ├── __init__.py
│   ├── sensor_node.py           # Multi-sensor telemetry and vibration generator
│   ├── feature_engineering.py   # RMS, Peak-to-Peak, Kurtosis, and FFT DSP
│   ├── tinyml_model.py          # PyTorch/NumPy TinyML MLP with INT8 PTQ
│   ├── edge_inference.py        # Microcontroller cycle-accurate inference engine
│   ├── mqtt_gateway.py          # ISO/IEC 20922 MQTT publisher and bandwidth calculator
│   └── pipeline.py              # End-to-end edge AIoT orchestration pipeline
├── tests/
│   ├── __init__.py
│   └── test_aiot_pipeline.py    # 6 comprehensive unit tests covering all components
├── wokwi/
│   ├── diagram.json             # Wokwi simulation circuit diagram (ESP32 + MPU6050 + DHT22)
│   └── sketch.ino               # C++ Arduino on-device firmware for ESP32
├── main.py                      # Main simulation entrypoint
├── train_model.py               # Complete training, evaluation, and quantization script
├── build_demo_video.py          # Automated 720p demo video generator with neural TTS
├── aiot_demo_video.mp4          # Final 3m44s HD presentation video (H.264 / AAC)
├── REPORT.md                    # Formal Academic Course Project Report
├── SOLUTIONS.md                 # Theoretical Study Guide & Chapter Solutions
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/Sais-gallego/developing-aiot.git
cd developing-aiot
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Simulation
```bash
python main.py
```

### 3. Retrain and Quantize the TinyML Model
```bash
python train_model.py
```

### 4. Run the Automated Test Suite
```bash
python -m unittest discover tests -v
```

---

## 📊 Benchmark & Quantitative Results

| Metric | Target Specification | Achieved Result | Evaluation Parameter |
|---|---|---|---|
| **Classification Accuracy** | $> 85.0\%$ | **100.0%** | Functionality (25%) |
| **Model Flash Footprint** | $< 4.0\text{ KB}$ | **120 bytes** (INT8) | Correctness (25%) |
| **SRAM Activation Buffer** | $< 1.0\text{ KB}$ | **23 bytes** | Correctness (25%) |
| **Average Inference Latency** | $< 1.0\text{ ms}$ | **~50.2 μs** | Functionality (25%) |
| **Bandwidth Reduction** | $> 75.0\%$ | **90.31%** | Functionality (25%) |
| **Test Suite Coverage** | $100\%$ passing | **6 / 6 passing** | Reproducibility (50%) |

---

## 📄 Formal Project Report

The complete academic written report is available at:
* 📄 [**REPORT.md**](REPORT.md)

---

## 🎥 Demonstration Video (Chapter 3: Demo)

The complete audio-visual demonstration addressing all course rubric criteria (sensing pipeline, Nyquist theorem, TinyML INT8 quantization, Pareto frontier, Wokwi simulation, and MQTT communication):
* **File:** [`aiot_demo_video.mp4`](aiot_demo_video.mp4) (Directly included in repository)
* **Duration:** 3 minutes 44 seconds (223.9 s) — *Strictly complies with the 3–5 min requirement*
* **Resolution:** 1280 x 720 (720p HD, 16:9 widescreen)
* **Codecs:** H.264 video @ 30 fps, AAC audio @ 192 kbps
* **File Size:** ~3.73 MB (< 500 MB limit)
* **Voiceover:** Spanish neural narration (`es-CO-GonzaloNeural`)

---

## 🤖 Declaración de Uso de Herramientas de IA (Mandatory Declaration)

En cumplimiento con los lineamientos de evaluación del curso **Developing the Artificial Intelligence of Things (2026 - 2027)**:
* **Herramienta de IA utilizada:** Google Antigravity / Gemini.
* **Modelo / Versión:** Gemini 2.5 / 3.0 Pro.
* **Propósito:** Asistencia en estructuración de código de pruebas unitarias (`tests/test_aiot_pipeline.py`), verificación del esquema JSON para Wokwi (`wokwi/diagram.json`) y maquetación de tablas comparativas de rendimiento en formato Markdown para el informe técnico.
* *El modelado físico de señales mecánicas, la arquitectura del perceptrón multicapa, los cálculos de cuantización simétrica INT8 y el análisis de optimización de ancho de banda fueron desarrollados y evaluados para demostrar la comprensión de los conceptos del curso.*

---

## 📜 License
MIT License. Developed for University of Helsinki MOOC.fi coursework by Sergio Andrés Ipuz Saavedra.
