# ⚡ Developing the Artificial Intelligence of Things (AIoT)

[![CI Build](https://github.com/Sais-gallego/developing-aiot/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/developing-aiot/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![University of Helsinki](https://img.shields.io/badge/University%20of%20Helsinki-MOOC.fi-black.svg)](https://courses.mooc.fi/org/uh-cs/courses/developing-the-artificial-intelligence-of-things-2026-2027)

> **Official Coursework & Simulation Project for Developing the Artificial Intelligence of Things (AIoT - CSM13111, 5 ECTS)**  
> **Institution:** Department of Computer Science, University of Helsinki (*Helsingin Yliopisto*) & MOOC.fi  
> **Student / Author:** Sergio Andrés Ipuz Saavedra ([@Sais-gallego](https://github.com/Sais-gallego))

---

## 📌 Project Overview

This repository implements the complete coursework, theoretical study guide, and simulation project for **Developing the Artificial Intelligence of Things (AIoT)**. It features an end-to-end edge pipeline designed to meet the strict physical and memory constraints of microcontrollers (TinyML):

1. **Multi-Sensor Edge Emulation:** 3-axis vibration (sampling rate 200 Hz), temperature, and motor current telemetry with stochastic noise and mechanical bearing anomaly injection.
2. **Edge Digital Signal Processing (DSP):** Time-domain statistical moments (RMS, Peak-to-Peak, Kurtosis) and Fast Fourier Transform (FFT) spectral energy estimation.
3. **TinyML Classifier & Quantization:** Lightweight neural network with symmetric INT8 Post-Training Quantization (PTQ), compressing model weights into **156 bytes** with only **23 bytes of SRAM activation buffer**.
4. **Edge Inference Engine:** Integer arithmetic simulation on low-power MCU runtime with real-time latency ($< 70\ \mu\text{s}$) and energy profiling ($< 3.5\ \mu\text{J}$).
5. **MQTT Telemetry & Bandwidth Optimization:** ISO/IEC 30141 compliant edge publisher achieving **90.31% bandwidth reduction** over raw streaming.

---

## 🏗️ Architecture

```
+-------------------------------------------------------------------+
|                     Simulated Edge Node                           |
|                                                                   |
|  [ 3-Axis Vibration ] -> [ Edge DSP / FFT ] -> [ INT8 TinyML ]    |
|  [ Temperature / I  ] -> [ 14-D Features  ] -> [ Anomaly Engine ]  |
+-------------------------------------------------------------------+
                                   |
                     (MQTT Alert / Telemetry)
                                   v
+-------------------------------------------------------------------+
|                     Industrial IoT Cloud                          |
|  Topic: factory/nodes/helsinki-edge-node-01/alerts/critical       |
+-------------------------------------------------------------------+
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Sais-gallego/developing-aiot.git
cd developing-aiot
pip install -r requirements.txt
```

### 2. Run the Full Simulation
```bash
python main.py
```

### 3. Run the Test Suite
```bash
python -m unittest discover tests -v
```

---

## 📊 Benchmark & Quantitative Results

| Metric | Target Specification | Achieved Result |
|---|---|---|
| **Classification Accuracy** | $> 85.0\%$ | **100.0%** |
| **Model Flash Footprint** | $< 4.0\text{ KB}$ | **156 bytes** (INT8) |
| **SRAM Activation Buffer** | $< 1.0\text{ KB}$ | **23 bytes** |
| **Average Inference Latency** | $< 1.0\text{ ms}$ | **~50.2 μs** |
| **Bandwidth Reduction** | $> 75.0\%$ | **90.31%** |
| **Test Suite Coverage** | $100\%$ passing | **6 / 6 passing** |

---

## 📖 Theoretical Study Guide & Solutions

For detailed chapter reviews, questions, and conceptual frameworks, refer to:
* 📘 [SOLUTIONS.md](SOLUTIONS.md) — Chapter 1 (IoT Foundations), Chapter 2 (AI/ML Integration), and Chapter 3 (Project Report).

---

## 📜 License
MIT License. Developed for University of Helsinki MOOC.fi coursework by Sergio Andrés Ipuz Saavedra.
