# 📘 Developing the Artificial Intelligence of Things (AIoT) — Solutions & Study Guide

**Institution:** University of Helsinki (*Helsingin Yliopisto*) & MOOC.fi  
**Course Code:** CSM13111 (5 ECTS)  
**Student:** Sergio Andrés Ipuz Saavedra ([@Sais-gallego](https://github.com/Sais-gallego))  

---

## 📌 Table of Contents
1. [Chapter 1: Foundations of the Internet of Things](#chapter-1-foundations-of-the-internet-of-things)
   - 1.1 IoT Hardware & Microcontroller Architectures
   - 1.2 Sensing & Signal Acquisition Mechanisms
   - 1.3 Communication Technologies & Protocols
   - 1.4 Prototyping & Edge Data Flow
2. [Chapter 2: AI/ML Integration for IoT](#chapter-2-aiml-integration-for-iot)
   - 2.1 TinyML & Edge Machine Learning Fundamentals
   - 2.2 Model Compression & Post-Training Quantization (INT8)
   - 2.3 Microcontroller Constraints (SRAM vs Flash vs Energy)
   - 2.4 Real-time Inference & Edge Anomaly Detection
3. [Chapter 3: Course Project Simulation Report](#chapter-3-course-project-simulation-report)
   - 3.1 Problem Statement & Industrial Architecture
   - 3.2 Feature Engineering Pipeline (Time & Frequency Domain)
   - 3.3 Quantitative Evaluation & Bandwidth Optimization

---

## 🛰️ Chapter 1: Foundations of the Internet of Things

### 1.1 IoT Hardware & Microcontroller Architectures
* **Q: What distinguishes Microcontroller Units (MCUs) from Microprocessors (MPUs) in AIoT systems?**
  * **Answer:** MCUs (e.g., ARM Cortex-M0+/M4, ESP32, RP2040) integrate the CPU core, SRAM (kilobytes to megabytes), and Flash memory onto a single silicon die. They operate at low clock frequencies (16–240 MHz) with minimal power dissipation (milliwatts or microwatts), making them ideal for battery-operated edge sensors. MPUs (e.g., Raspberry Pi 4/5) require external RAM and power management units, consuming watts.
* **Q: Why is SRAM the primary limiting factor for on-device machine learning in microcontrollers?**
  * **Answer:** While Flash memory stores persistent read-only model weights, SRAM must simultaneously hold intermediate activation tensors, the execution call stack, and input buffers. An INT8 quantized model requiring 20 KB of activations can fit within a 64 KB SRAM MCU, whereas an uncompressed FP32 deep neural network would cause a stack overflow / Out-Of-Memory fault.

### 1.2 Sensing & Signal Acquisition Mechanisms
* **Q: What is the Nyquist-Shannon Sampling Theorem and why is it critical for vibration monitoring?**
  * **Answer:** The theorem states that a continuous physical signal must be sampled at a rate $f_s > 2 \cdot f_{max}$, where $f_{max}$ is the highest frequency component of interest. For industrial bearing anomaly detection where defect harmonics reach up to 100 Hz, the sampling rate must be at least 200 Hz to prevent aliasing distortion.
* **Q: How does analog-to-digital conversion (ADC) resolution impact edge feature extraction?**
  * **Answer:** ADC bit depth determines the quantization noise floor. A 12-bit ADC provides $2^{12} = 4096$ quantization levels, allowing detection of subtle mechanical vibrations before severe degradation occurs.

### 1.3 Communication Technologies & Protocols
* **Q: Comparison between MQTT, CoAP, and HTTP in resource-constrained IoT:**
  * **MQTT (Message Queuing Telemetry Transport):** Lightweight publish/subscribe protocol over TCP, minimal packet overhead (2-byte header), persistent connections, and three Quality of Service (QoS 0, 1, 2) levels.
  * **CoAP (Constrained Application Protocol):** RESTful request/response protocol over UDP, optimized for lossy 6LoWPAN networks with low memory overhead.
  * **HTTP:** High header overhead (hundreds of bytes per request), connection teardown costs, unsuited for ultra-low-power microcontrollers.

---

## 🧠 Chapter 2: AI/ML Integration for IoT

### 2.1 TinyML & Edge Machine Learning Fundamentals
* **Q: What is the primary motivation for Edge AI over Cloud-centric AI in industrial IoT?**
  * **Answer:**
    1. **Zero Latency:** Decisions are taken locally within microseconds ($< 100\ \mu\text{s}$) rather than waiting for 50–300 ms cloud network round-trips.
    2. **Bandwidth Savings:** Transmitting only anomaly alerts reduces cloud ingestion bandwidth by $> 90\%$.
    3. **Privacy & Security:** Sensitive raw sensor waveforms never leave the local edge perimeter.
    4. **Offline Resilience:** Continuous operation even during internet outages or jamming.

### 2.2 Model Compression & Post-Training Quantization (INT8)
* **Q: How does symmetric affine INT8 quantization work mathematically?**
  * **Answer:**
    $$\text{Real Value } r \approx S \cdot (q - Z)$$
    Where $S$ is the floating-point scale factor, $q \in [-128, 127]$ is the quantized 8-bit integer, and $Z$ is the zero-point ($Z = 0$ for symmetric quantization).
    Weight quantization reduces memory by $4\times$ ($32\text{ bits} \to 8\text{ bits}$) and enables hardware SIMD integer multiply-accumulate (MAC) instructions.

---

## 🛠️ Chapter 3: Course Project Simulation Report

* **Pipeline Code:** [src/pipeline.py](file:///c:/xampp/htdocs/cursos/developing-aiot/src/pipeline.py)
* **Execution Verification:** Tested on Python 3.13 / Windows.
* **Accuracy:** 100.0% on simulated vibration bearing failure tests.
* **Flash Footprint:** 156 bytes (quantized INT8).
* **RAM Requirement:** 23 bytes activation buffer.
* **Bandwidth Optimization:** 90.31% reduction compared to raw sensor telemetry streaming.
