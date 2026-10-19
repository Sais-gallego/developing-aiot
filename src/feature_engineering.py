"""
Feature Engineering Module for Edge AIoT Digital Signal Processing (DSP).
Pure NumPy implementation for ultra-lightweight edge and microcontroller environments.
"""

from typing import Dict, List
import numpy as np
from .sensor_node import SensorReading


class EdgeFeatureExtractor:
    """
    Lightweight feature extractor designed to run under embedded resource constraints.
    """

    def __init__(self, sampling_rate_hz: int = 200):
        self.sampling_rate_hz = sampling_rate_hz

    def extract_time_features(self, signal: np.ndarray) -> List[float]:
        """
        Extract statistical time-domain features using pure NumPy:
        - Mean
        - Standard Deviation
        - RMS (Root Mean Square)
        - Peak-to-Peak
        - Kurtosis (bearing wear indicator: 4th standardized moment)
        """
        mean_val = float(np.mean(signal))
        std_val = float(np.std(signal))
        rms_val = float(np.sqrt(np.mean(signal**2)))
        ptp_val = float(np.ptp(signal))

        # 4th standardized moment (Kurtosis)
        std_safe = max(std_val, 1e-6)
        kurt_val = float(np.mean(((signal - mean_val) / std_safe) ** 4) - 3.0)

        return [mean_val, std_val, rms_val, ptp_val, kurt_val]

    def extract_frequency_features(self, signal: np.ndarray) -> List[float]:
        """
        Extract spectral features via discrete FFT:
        - Dominant peak frequency
        - Spectral energy in high frequency band (>50 Hz)
        - Total spectral energy
        """
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate_hz)

        # Peak frequency (ignoring DC at 0 Hz)
        peak_idx = int(np.argmax(fft_power[1:])) + 1
        peak_freq = float(freqs[peak_idx])

        # High frequency energy (> 50 Hz)
        hf_mask = freqs > 50.0
        hf_energy = float(np.sum(fft_power[hf_mask]))
        total_energy = float(np.sum(fft_power[1:]))

        hf_ratio = hf_energy / (total_energy + 1e-8)

        return [peak_freq, total_energy, hf_ratio]

    def process_reading(self, reading: SensorReading) -> np.ndarray:
        """
        Construct a compact 14-dimensional feature vector:
        - Vibration X: 3 time features (RMS, Std, PTP) + 2 freq features (Peak, HF ratio) = 5
        - Vibration Y: 3 time features (RMS, Std, PTP) = 3
        - Vibration Z: 3 time features (RMS, Std, PTP) + 1 freq feature (HF ratio) = 4
        - Temperature = 1
        - Current = 1
        Total = 14 float features.
        """
        feats_x_time = self.extract_time_features(reading.vibration_x)
        feats_x_freq = self.extract_frequency_features(reading.vibration_x)

        feats_y_time = self.extract_time_features(reading.vibration_y)
        feats_z_time = self.extract_time_features(reading.vibration_z)
        feats_z_freq = self.extract_frequency_features(reading.vibration_z)

        vector = [
            feats_x_time[1],  # X Std
            feats_x_time[2],  # X RMS
            feats_x_time[3],  # X PTP
            feats_x_freq[0],  # X Peak Freq
            feats_x_freq[2],  # X HF Ratio
            feats_y_time[1],  # Y Std
            feats_y_time[2],  # Y RMS
            feats_y_time[3],  # Y PTP
            feats_z_time[1],  # Z Std
            feats_z_time[2],  # Z RMS
            feats_z_time[3],  # Z PTP
            feats_z_freq[2],  # Z HF Ratio
            reading.temperature_celsius,
            reading.current_ampere,
        ]

        return np.array(vector, dtype=np.float32)
