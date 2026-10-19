"""
Sensor Node Emulation Module for AIoT Edge Systems.
Simulates multi-sensor physical data acquisition: 3-axis vibration, temperature, and current.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SensorReading:
    timestamp: float
    vibration_x: np.ndarray
    vibration_y: np.ndarray
    vibration_z: np.ndarray
    temperature_celsius: float
    current_ampere: float
    is_anomaly: bool


class SimulatedSensorNode:
    """
    Simulates an industrial edge sensor node (e.g., STM32/ESP32 with IMU & telemetry).
    """

    def __init__(
        self,
        node_id: str = "aiot-node-01",
        sampling_rate_hz: int = 200,
        window_duration_sec: float = 1.0,
        baseline_temp: float = 42.0,
        baseline_current: float = 4.5,
        random_seed: Optional[int] = 42,
    ):
        self.node_id = node_id
        self.sampling_rate_hz = sampling_rate_hz
        self.window_duration_sec = window_duration_sec
        self.num_samples = int(sampling_rate_hz * window_duration_sec)
        self.baseline_temp = baseline_temp
        self.baseline_current = baseline_current
        self.rng = np.random.default_rng(random_seed)
        self.current_time = 0.0

    def sample_window(self, inject_anomaly: bool = False) -> SensorReading:
        """
        Generate a synchronized window of sensor samples.
        """
        t = np.linspace(
            self.current_time,
            self.current_time + self.window_duration_sec,
            self.num_samples,
            endpoint=False,
        )

        if not inject_anomaly:
            # Baseline normal vibration: low amplitude 25 Hz fundamental harmonic + noise
            vx = 0.5 * np.sin(2 * np.pi * 25 * t) + self.rng.normal(0, 0.1, self.num_samples)
            vy = 0.4 * np.sin(2 * np.pi * 25 * t + np.pi / 4) + self.rng.normal(0, 0.1, self.num_samples)
            vz = 0.6 * np.cos(2 * np.pi * 25 * t) + self.rng.normal(0, 0.1, self.num_samples)
            temp = self.baseline_temp + float(self.rng.normal(0, 0.3))
            curr = self.baseline_current + float(self.rng.normal(0, 0.05))
        else:
            # Anomaly: Bearing defect introduces high-frequency spikes (75 Hz + 110 Hz) and elevated temperature
            vx = (
                1.8 * np.sin(2 * np.pi * 75 * t)
                + 1.2 * np.sin(2 * np.pi * 110 * t)
                + self.rng.normal(0, 0.35, self.num_samples)
            )
            vy = 1.5 * np.sin(2 * np.pi * 75 * t + np.pi / 3) + self.rng.normal(0, 0.3, self.num_samples)
            vz = 2.0 * np.cos(2 * np.pi * 110 * t) + self.rng.normal(0, 0.4, self.num_samples)
            temp = self.baseline_temp + float(self.rng.uniform(15.0, 28.0))
            curr = self.baseline_current + float(self.rng.uniform(1.8, 3.5))

        self.current_time += self.window_duration_sec

        return SensorReading(
            timestamp=self.current_time,
            vibration_x=vx,
            vibration_y=vy,
            vibration_z=vz,
            temperature_celsius=temp,
            current_ampere=curr,
            is_anomaly=inject_anomaly,
        )
