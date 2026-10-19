"""
IoT Gateway & MQTT Telemetry Module.
Formats edge inference results into standard IoT telemetry payloads and optimizes bandwidth.
"""

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Dict, List, Optional
from .edge_inference import EdgeInferenceResult
from .sensor_node import SensorReading


@dataclass
class MqttTelemetryPacket:
    topic: str
    qos: int
    payload_json: str
    is_alert: bool


class EdgeMqttPublisher:
    """
    Simulated MQTT Edge Client (e.g. Eclipse Paho / MQTT v5.0).
    Applies Edge-Filtering: transmits lightweight heartbeats during normal states,
    and priority instant alert packets with sensor snapshots upon anomaly detection.
    """

    def __init__(self, node_id: str = "aiot-edge-01", broker_host: str = "mqtt.factory-iot.net"):
        self.node_id = node_id
        self.broker_host = broker_host
        self.published_history: List[MqttTelemetryPacket] = []

    def create_packet(
        self,
        reading: SensorReading,
        inference: EdgeInferenceResult,
        qos: int = 1,
    ) -> MqttTelemetryPacket:
        """
        Build an edge telemetry JSON packet.
        """
        topic = f"factory/nodes/{self.node_id}/telemetry"
        if inference.is_anomaly:
            topic = f"factory/nodes/{self.node_id}/alerts/critical"

        payload = {
            "node_id": self.node_id,
            "timestamp": reading.timestamp,
            "status": inference.predicted_class,
            "anomaly_probability": round(inference.anomaly_probability, 4),
            "telemetry": {
                "temperature_c": round(reading.temperature_celsius, 2),
                "current_a": round(reading.current_ampere, 2),
            },
            "edge_metrics": {
                "latency_us": round(inference.latency_microseconds, 1),
                "energy_uj": round(inference.mcu_energy_microjoules, 3),
            },
        }

        packet = MqttTelemetryPacket(
            topic=topic,
            qos=qos,
            payload_json=json.dumps(payload, indent=None),
            is_alert=inference.is_anomaly,
        )

        self.published_history.append(packet)
        return packet
