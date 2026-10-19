"""
TinyML Neural Network Model for Edge Anomaly Detection.
Supports model training, post-training INT8 quantization, and microcontroller footprint estimation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class QuantizedLayer:
    weights_int8: np.ndarray
    bias_int32: np.ndarray
    scale_w: float
    scale_in: float
    scale_out: float


class TinyMLClassifier:
    """
    Lightweight 2-layer Neural Network designed for microcontroller deployment.
    Architecture: 14 inputs -> 8 hidden neurons (ReLU) -> 1 output neuron (Sigmoid).
    Total parameters: (14*8 + 8) + (8*1 + 1) = 120 + 9 = 129 parameters (< 1 KB footprint).
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 8, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights (He initialization)
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        self.quantized_l1: Optional[QuantizedLayer] = None
        self.quantized_l2: Optional[QuantizedLayer] = None

        # Normalization parameters (mean & std from calibration)
        self.feature_means = np.zeros(input_dim, dtype=np.float32)
        self.feature_stds = np.ones(input_dim, dtype=np.float32)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 150, lr: float = 0.05) -> Dict[str, float]:
        """
        Train the lightweight model using batch gradient descent.
        """
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.feature_means) / self.feature_stds

        m = X.shape[0]
        y_true = y.reshape(-1, 1)

        final_loss = 0.0
        for _ in range(epochs):
            # Forward pass
            z1 = np.dot(X_norm, self.W1) + self.b1
            a1 = self.relu(z1)

            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)

            # Binary Cross-Entropy loss
            loss = -np.mean(y_true * np.log(a2 + 1e-8) + (1 - y_true) * np.log(1 - a2 + 1e-8))
            final_loss = float(loss)

            # Backward pass
            dz2 = (a2 - y_true) / m
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (z1 > 0)
            dW1 = np.dot(X_norm.T, dz1)
            db1 = np.sum(dz1, axis=0)

            # Gradient descent update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        return {"final_loss": final_loss}

    def predict_proba_fp32(self, x: np.ndarray) -> float:
        """
        Float32 reference forward inference.
        """
        x_norm = (x - self.feature_means) / self.feature_stds
        z1 = np.dot(x_norm, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return float(a2[0])

    def quantize_to_int8(self) -> Tuple[QuantizedLayer, QuantizedLayer]:
        """
        Post-Training Quantization (symmetric INT8 range [-127, 127]).
        """
        # Layer 1 quantization
        max_w1 = float(np.max(np.abs(self.W1)))
        scale_w1 = max_w1 / 127.0 if max_w1 > 0 else 1.0
        w1_int8 = np.clip(np.round(self.W1 / scale_w1), -127, 127).astype(np.int8)
        b1_int32 = np.round(self.b1 / (scale_w1 * 0.1)).astype(np.int32)

        self.quantized_l1 = QuantizedLayer(
            weights_int8=w1_int8,
            bias_int32=b1_int32,
            scale_w=scale_w1,
            scale_in=0.1,
            scale_out=0.1,
        )

        # Layer 2 quantization
        max_w2 = float(np.max(np.abs(self.W2)))
        scale_w2 = max_w2 / 127.0 if max_w2 > 0 else 1.0
        w2_int8 = np.clip(np.round(self.W2 / scale_w2), -127, 127).astype(np.int8)
        b2_int32 = np.round(self.b2 / (scale_w2 * 0.1)).astype(np.int32)

        self.quantized_l2 = QuantizedLayer(
            weights_int8=w2_int8,
            bias_int32=b2_int32,
            scale_w=scale_w2,
            scale_in=0.1,
            scale_out=1.0,
        )

        return self.quantized_l1, self.quantized_l2

    def estimate_hardware_footprint(self) -> Dict[str, float]:
        """
        Calculate memory footprint for microcontroller Flash (ROM) and SRAM (RAM).
        """
        total_weights_count = self.W1.size + self.W2.size
        total_biases_count = self.b1.size + self.b2.size

        # FP32 footprint
        fp32_bytes = (total_weights_count + total_biases_count) * 4

        # INT8 quantized footprint: weights (1 byte each) + biases (4 bytes int32)
        int8_bytes = (total_weights_count * 1) + (total_biases_count * 4)

        # Activation buffer in SRAM (input buffer + hidden layer buffer + output buffer)
        sram_activation_buffer_bytes = (self.input_dim + self.hidden_dim + self.output_dim) * 1

        return {
            "parameters_count": float(total_weights_count + total_biases_count),
            "fp32_model_size_bytes": float(fp32_bytes),
            "int8_model_size_bytes": float(int8_bytes),
            "compression_ratio": float(fp32_bytes / max(int8_bytes, 1)),
            "sram_buffer_bytes": float(sram_activation_buffer_bytes),
        }
