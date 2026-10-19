"""
Edge Inference Runtime Module.
Simulates on-device TinyML execution (INT8 integer arithmetic) on an ultra-low-power MCU.
"""

from dataclasses import dataclass
import time
from typing import Dict, Tuple
import numpy as np
from .tinyml_model import TinyMLClassifier


@dataclass
class EdgeInferenceResult:
    predicted_class: str
    anomaly_probability: float
    is_anomaly: bool
    latency_microseconds: float
    mcu_energy_microjoules: float


class EdgeInferenceEngine:
    """
    Simulates embedded inference running on an ARM Cortex-M4 / ESP32 microcontroller (e.g. 80 MHz clock).
    """

    def __init__(self, model: TinyMLClassifier, anomaly_threshold: float = 0.5, mcu_power_milliwatts: float = 45.0):
        self.model = model
        self.anomaly_threshold = anomaly_threshold
        self.mcu_power_milliwatts = mcu_power_milliwatts

        # Ensure model is quantized
        if self.model.quantized_l1 is None:
            self.model.quantize_to_int8()

    def run_inference(self, feature_vector: np.ndarray) -> EdgeInferenceResult:
        """
        Execute simulated INT8 quantized inference and measure edge execution metrics.
        """
        start_ns = time.perf_counter_ns()

        # Step 1: Normalize input to calibrated INT8 scale
        x_norm = (feature_vector - self.model.feature_means) / self.model.feature_stds
        x_int8 = np.clip(np.round(x_norm / self.model.quantized_l1.scale_in), -128, 127).astype(np.int8)

        # Step 2: Layer 1 INT8 matrix-vector multiplication & ReLU
        l1 = self.model.quantized_l1
        # Integer dot product
        z1_int32 = np.dot(x_int8.astype(np.int32), l1.weights_int8.astype(np.int32)) + l1.bias_int32
        # Dequantize & ReLU
        a1_fp = np.maximum(0.0, z1_int32.astype(np.float32) * (l1.scale_w * l1.scale_in))

        # Step 3: Layer 2 computation & Sigmoid
        l2 = self.model.quantized_l2
        a1_int8 = np.clip(np.round(a1_fp / l2.scale_in), -128, 127).astype(np.int8)
        z2_int32 = np.dot(a1_int8.astype(np.int32), l2.weights_int8.astype(np.int32)) + l2.bias_int32
        z2_fp = z2_int32.astype(np.float32) * (l2.scale_w * l2.scale_in)

        prob = 1.0 / (1.0 + np.exp(-np.clip(z2_fp[0], -15.0, 15.0)))
        is_anomaly = bool(prob >= self.anomaly_threshold)

        duration_ns = time.perf_counter_ns() - start_ns
        latency_us = max(float(duration_ns / 1000.0), 12.5)  # realistic MCU latency floor ~12.5 microseconds

        # Energy = Power (mW) * time (us) / 1000 = microjoules (uJ)
        energy_uj = (self.mcu_power_milliwatts * latency_us) / 1000.0

        return EdgeInferenceResult(
            predicted_class="ANOMALY_BEARING_FAULT" if is_anomaly else "NORMAL_OPERATION",
            anomaly_probability=float(prob),
            is_anomaly=is_anomaly,
            latency_microseconds=latency_us,
            mcu_energy_microjoules=energy_uj,
        )
