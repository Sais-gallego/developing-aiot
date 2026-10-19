"""
End-to-End AIoT Simulation Pipeline Runner.
Orchestrates sensor acquisition, edge DSP, quantized inference, and IoT cloud telemetry.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .edge_inference import EdgeInferenceEngine, EdgeInferenceResult
from .feature_engineering import EdgeFeatureExtractor
from .mqtt_gateway import EdgeMqttPublisher, MqttTelemetryPacket
from .sensor_node import SimulatedSensorNode
from .tinyml_model import TinyMLClassifier


class AioTPipeline:
    """
    Main orchestration class for the AIoT Edge Simulation Course Project.
    """

    def __init__(self, node_id: str = "aiot-unit-42"):
        self.node_id = node_id
        self.sensor = SimulatedSensorNode(node_id=node_id, sampling_rate_hz=200, window_duration_sec=1.0)
        self.extractor = EdgeFeatureExtractor(sampling_rate_hz=200)
        self.model = TinyMLClassifier()
        self.inference_engine: Optional[EdgeInferenceEngine] = None
        self.mqtt_publisher = EdgeMqttPublisher(node_id=node_id)

    def train_and_calibrate(self, num_samples_per_class: int = 100) -> Dict[str, float]:
        """
        Generate synthetic calibration dataset, extract features, and train the model.
        """
        X_list = []
        y_list = []

        # Normal samples
        for _ in range(num_samples_per_class):
            reading = self.sensor.sample_window(inject_anomaly=False)
            feats = self.extractor.process_reading(reading)
            X_list.append(feats)
            y_list.append(0)

        # Anomaly samples
        for _ in range(num_samples_per_class):
            reading = self.sensor.sample_window(inject_anomaly=True)
            feats = self.extractor.process_reading(reading)
            X_list.append(feats)
            y_list.append(1)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Shuffle
        indices = np.arange(len(y))
        np.random.default_rng(42).shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Fit model
        train_stats = self.model.fit(X, y, epochs=180, lr=0.06)

        # Quantize and initialize runtime engine
        self.model.quantize_to_int8()
        self.inference_engine = EdgeInferenceEngine(self.model, anomaly_threshold=0.5)

        return train_stats

    def run_simulation(self, total_windows: int = 20, anomaly_frequency: int = 5) -> Dict[str, Any]:
        """
        Run continuous edge simulation windows.
        """
        if self.inference_engine is None:
            self.train_and_calibrate()

        results: List[Dict[str, Any]] = []
        correct_predictions = 0

        total_raw_bytes = 0
        total_edge_published_bytes = 0

        for i in range(total_windows):
            inject_anomaly = (i > 0 and i % anomaly_frequency == 0)
            reading = self.sensor.sample_window(inject_anomaly=inject_anomaly)
            features = self.extractor.process_reading(reading)
            inference = self.inference_engine.run_inference(features)

            packet = self.mqtt_publisher.create_packet(reading, inference)

            # Raw byte estimate (uncompressed sensor waveform streaming: 200 samples * 3 axes * 4 bytes + temp + curr)
            raw_window_bytes = (len(reading.vibration_x) * 3 * 4) + 8
            total_raw_bytes += raw_window_bytes
            total_edge_published_bytes += len(packet.payload_json.encode("utf-8"))

            is_correct = (inference.is_anomaly == inject_anomaly)
            if is_correct:
                correct_predictions += 1

            results.append({
                "window": i + 1,
                "ground_truth_anomaly": inject_anomaly,
                "predicted_anomaly": inference.is_anomaly,
                "confidence": inference.anomaly_probability,
                "latency_us": inference.latency_microseconds,
                "energy_uj": inference.mcu_energy_microjoules,
                "mqtt_topic": packet.topic,
            })

        accuracy = float(correct_predictions / total_windows)
        bandwidth_saved_pct = float((1.0 - (total_edge_published_bytes / total_raw_bytes)) * 100.0)
        hw_footprint = self.model.estimate_hardware_footprint()

        return {
            "total_windows": total_windows,
            "accuracy": accuracy,
            "bandwidth_saved_percentage": bandwidth_saved_pct,
            "hardware_footprint": hw_footprint,
            "simulation_details": results,
        }
