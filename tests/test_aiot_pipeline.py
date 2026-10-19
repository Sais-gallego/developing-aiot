"""
Unit and Integration Test Suite for Developing AIoT Course Project.
Compatible with both standard unittest and pytest.
"""

import json
import unittest
import numpy as np
from src.edge_inference import EdgeInferenceEngine
from src.feature_engineering import EdgeFeatureExtractor
from src.mqtt_gateway import EdgeMqttPublisher
from src.pipeline import AioTPipeline
from src.sensor_node import SimulatedSensorNode
from src.tinyml_model import TinyMLClassifier


class TestAioTPipeline(unittest.TestCase):

    def test_sensor_node_normal_vs_anomaly(self):
        node = SimulatedSensorNode(sampling_rate_hz=200, window_duration_sec=1.0)
        normal = node.sample_window(inject_anomaly=False)
        anomaly = node.sample_window(inject_anomaly=True)

        self.assertEqual(len(normal.vibration_x), 200)
        self.assertFalse(normal.is_anomaly)
        self.assertTrue(anomaly.is_anomaly)
        self.assertGreater(anomaly.temperature_celsius, normal.temperature_celsius)
        self.assertGreater(np.std(anomaly.vibration_x), np.std(normal.vibration_x))

    def test_feature_extractor_dimensions(self):
        extractor = EdgeFeatureExtractor(sampling_rate_hz=200)
        node = SimulatedSensorNode()
        reading = node.sample_window(inject_anomaly=False)

        features = extractor.process_reading(reading)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (14,))
        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isinf(features).any())

    def test_tinyml_model_training_and_quantization(self):
        model = TinyMLClassifier()
        X = np.random.randn(50, 14).astype(np.float32)
        y = np.array([0] * 25 + [1] * 25, dtype=np.float32)

        stats = model.fit(X, y, epochs=50, lr=0.05)
        self.assertIn("final_loss", stats)
        self.assertGreaterEqual(stats["final_loss"], 0.0)

        l1, l2 = model.quantize_to_int8()
        self.assertEqual(l1.weights_int8.dtype, np.int8)
        self.assertEqual(l2.weights_int8.dtype, np.int8)

        footprint = model.estimate_hardware_footprint()
        self.assertLess(footprint["int8_model_size_bytes"], footprint["fp32_model_size_bytes"])
        self.assertGreater(footprint["compression_ratio"], 1.5)

    def test_edge_inference_engine(self):
        model = TinyMLClassifier()
        engine = EdgeInferenceEngine(model)
        fake_feats = np.random.randn(14).astype(np.float32)

        result = engine.run_inference(fake_feats)
        self.assertGreaterEqual(result.anomaly_probability, 0.0)
        self.assertLessEqual(result.anomaly_probability, 1.0)
        self.assertGreater(result.latency_microseconds, 0.0)
        self.assertGreater(result.mcu_energy_microjoules, 0.0)
        self.assertIn(result.predicted_class, ["NORMAL_OPERATION", "ANOMALY_BEARING_FAULT"])

    def test_mqtt_gateway_alert_routing(self):
        publisher = EdgeMqttPublisher(node_id="test-node-99")
        node = SimulatedSensorNode()
        model = TinyMLClassifier()
        engine = EdgeInferenceEngine(model)

        normal_reading = node.sample_window(inject_anomaly=False)
        normal_inf = engine.run_inference(np.zeros(14, dtype=np.float32))
        normal_inf.is_anomaly = False
        normal_inf.predicted_class = "NORMAL_OPERATION"
        pkt_normal = publisher.create_packet(normal_reading, normal_inf)
        self.assertEqual(pkt_normal.topic, "factory/nodes/test-node-99/telemetry")

        normal_inf.is_anomaly = True
        normal_inf.predicted_class = "ANOMALY_BEARING_FAULT"
        pkt_alert = publisher.create_packet(normal_reading, normal_inf)
        self.assertEqual(pkt_alert.topic, "factory/nodes/test-node-99/alerts/critical")
        data = json.loads(pkt_alert.payload_json)
        self.assertEqual(data["status"], "ANOMALY_BEARING_FAULT")

    def test_end_to_end_pipeline_simulation(self):
        pipeline = AioTPipeline(node_id="aiot-test-runner")
        sim_stats = pipeline.run_simulation(total_windows=20, anomaly_frequency=4)

        self.assertEqual(sim_stats["total_windows"], 20)
        self.assertGreaterEqual(sim_stats["accuracy"], 0.85)
        self.assertGreater(sim_stats["bandwidth_saved_percentage"], 70.0)
        self.assertEqual(len(sim_stats["simulation_details"]), 20)


if __name__ == "__main__":
    unittest.main()
