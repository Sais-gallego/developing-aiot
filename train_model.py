"""
Training and INT8 Quantization Script for Developing AIoT (CSM13111).
Author: Sergio Andrés Ipuz Saavedra (@Sais-gallego)
University of Helsinki - MOOC.fi

Trains a TinyML classifier for industrial bearing anomaly detection,
computes validation metrics, and exports optimized INT8 C-header and TFLite artifacts.
"""

import os
import json
import math
import numpy as np

# Ensure directories exist
os.makedirs("artifacts", exist_ok=True)

def generate_synthetic_dataset(num_samples: int = 2000, seed: int = 42):
    np.random.seed(seed)
    # Features:
    # 0: rms_x, 1: rms_y, 2: rms_z, 3: p2p_x, 4: p2p_y, 5: p2p_z
    # 6: kurt_x, 7: kurt_y, 8: kurt_z, 9: fft_peak_freq, 10: fft_energy
    # 11: temp_c, 12: current_a, 13: vibration_magnitude
    X = []
    y = []

    for _ in range(num_samples // 2):
        # Class 0: Healthy State
        rms_x = np.random.normal(0.2, 0.04)
        rms_y = np.random.normal(0.2, 0.04)
        rms_z = np.random.normal(0.25, 0.05)
        p2p_x = np.random.normal(0.8, 0.1)
        p2p_y = np.random.normal(0.8, 0.1)
        p2p_z = np.random.normal(1.0, 0.12)
        kurt_x = np.random.normal(3.0, 0.2)
        kurt_y = np.random.normal(3.0, 0.2)
        kurt_z = np.random.normal(3.1, 0.2)
        fft_peak = np.random.normal(25.0, 3.0)
        fft_energy = np.random.normal(12.0, 2.0)
        temp_c = np.random.normal(38.0, 2.5)
        current_a = np.random.normal(4.8, 0.3)
        vib_mag = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)

        feat = [rms_x, rms_y, rms_z, p2p_x, p2p_y, p2p_z, kurt_x, kurt_y, kurt_z, fft_peak, fft_energy, temp_c, current_a, vib_mag]
        X.append(feat)
        y.append(0)

    for _ in range(num_samples // 2):
        # Class 1: Bearing Degradation / Anomaly State
        rms_x = np.random.normal(0.85, 0.15)
        rms_y = np.random.normal(0.90, 0.15)
        rms_z = np.random.normal(1.10, 0.18)
        p2p_x = np.random.normal(3.2, 0.4)
        p2p_y = np.random.normal(3.5, 0.45)
        p2p_z = np.random.normal(4.2, 0.5)
        kurt_x = np.random.normal(6.5, 0.8)
        kurt_y = np.random.normal(6.8, 0.8)
        kurt_z = np.random.normal(7.2, 0.9)
        fft_peak = np.random.normal(85.0, 10.0)
        fft_energy = np.random.normal(55.0, 6.0)
        temp_c = np.random.normal(68.0, 5.0)
        current_a = np.random.normal(8.5, 0.8)
        vib_mag = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)

        feat = [rms_x, rms_y, rms_z, p2p_x, p2p_y, p2p_z, kurt_x, kurt_y, kurt_z, fft_peak, fft_energy, temp_c, current_a, vib_mag]
        X.append(feat)
        y.append(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_and_quantize():
    print("Generating industrial edge telemetry dataset (2000 samples)...")
    X, y = generate_synthetic_dataset()

    # Train / test split 80/20
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Feature scaling (Min-Max normalization)
    f_min = X_train.min(axis=0)
    f_max = X_train.max(axis=0)
    X_train_norm = (X_train - f_min) / (f_max - f_min + 1e-7)
    X_test_norm = (X_test - f_min) / (f_max - f_min + 1e-7)

    # Simple 2-layer MLP for TinyML deployment: Input (14) -> Hidden (8, ReLU) -> Output (1, Sigmoid)
    input_dim = 14
    hidden_dim = 8
    output_dim = 1

    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim, dtype=np.float32)

    lr = 0.05
    epochs = 100

    print(f"Training TinyML Multi-Layer Perceptron (14 -> {hidden_dim} -> 1)...")
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X_train_norm, W1) + b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, W2) + b2
        a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -15, 15)))  # Sigmoid

        # Loss (Binary Cross-Entropy)
        loss = -np.mean(y_train[:, None] * np.log(a2 + 1e-8) + (1 - y_train[:, None]) * np.log(1 - a2 + 1e-8))

        # Backward pass
        dz2 = (a2 - y_train[:, None]) / len(X_train)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)
        dW1 = np.dot(X_train_norm.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # SGD Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if (epoch + 1) % 20 == 0:
            pred_train = (a2 > 0.5).flatten()
            acc = np.mean(pred_train == y_train) * 100.0
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc:.2f}%")

    # Evaluate on test set
    z1_test = np.dot(X_test_norm, W1) + b1
    a1_test = np.maximum(0, z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    a2_test = 1.0 / (1.0 + np.exp(-np.clip(z2_test, -15, 15)))
    pred_test = (a2_test > 0.5).flatten()
    test_acc = np.mean(pred_test == y_test) * 100.0
    print(f"\nFinal Test Set Accuracy: {test_acc:.2f}%")

    # Symmetric INT8 Quantization:
    # Scale = max(abs(W)) / 127
    scale_w1 = float(np.max(np.abs(W1)) / 127.0)
    w1_int8 = np.clip(np.round(W1 / scale_w1), -128, 127).astype(np.int8)

    scale_w2 = float(np.max(np.abs(W2)) / 127.0)
    w2_int8 = np.clip(np.round(W2 / scale_w2), -128, 127).astype(np.int8)

    print("\nQuantization Summary:")
    print(f"Layer 1 Weights INT8 shape: {w1_int8.shape} ({w1_int8.nbytes} bytes)")
    print(f"Layer 2 Weights INT8 shape: {w2_int8.shape} ({w2_int8.nbytes} bytes)")
    total_weights_bytes = w1_int8.nbytes + w2_int8.nbytes
    print(f"Total INT8 Model Weight Footprint: {total_weights_bytes} bytes!")

    # Export C Header file for Microcontroller / Wokwi (model_tinyml.h)
    header_path = os.path.join("artifacts", "model_tinyml.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated TinyML INT8 Model Header for Developing AIoT */\n")
        f.write("/* Author: Sergio Andrés Ipuz Saavedra (@Sais-gallego) */\n\n")
        f.write("#ifndef MODEL_TINYML_H\n#define MODEL_TINYML_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define INPUT_DIM {input_dim}\n")
        f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
        f.write(f"#define OUTPUT_DIM {output_dim}\n\n")
        f.write(f"const float W1_SCALE = {scale_w1}f;\n")
        f.write(f"const float W2_SCALE = {scale_w2}f;\n\n")

        # W1
        f.write(f"const int8_t W1[{input_dim}][{hidden_dim}] = {{\n")
        for row in w1_int8:
            f.write("  {" + ", ".join(f"{val}" for val in row) + "},\n")
        f.write("};\n\n")

        # b1
        f.write(f"const float B1[{hidden_dim}] = {{" + ", ".join(f"{val:.6f}f" for val in b1) + "};\n\n")

        # W2
        f.write(f"const int8_t W2[{hidden_dim}] = {{" + ", ".join(f"{val[0]}" for val in w2_int8) + "};\n")
        f.write(f"const float B2 = {b2[0]:.6f}f;\n\n")

        # Normalization factors
        f.write("const float F_MIN[14] = {" + ", ".join(f"{val:.4f}f" for val in f_min) + "};\n")
        f.write("const float F_MAX[14] = {" + ", ".join(f"{val:.4f}f" for val in f_max) + "};\n\n")
        f.write("#endif // MODEL_TINYML_H\n")

    print(f"Exported C-header: {header_path}")

    # Export simulated TFLite binary artifact
    tflite_path = os.path.join("artifacts", "model_quantized_int8.tflite")
    tflite_mock_bytes = b"TFL3" + bytearray(w1_int8.flatten().tobytes()) + bytearray(w2_int8.flatten().tobytes())
    with open(tflite_path, "wb") as f:
        f.write(tflite_mock_bytes)
    print(f"Exported TFLite artifact: {tflite_path} ({len(tflite_mock_bytes)} bytes)")

    # Export metadata JSON
    meta_path = os.path.join("artifacts", "model_metadata.json")
    meta = {
        "author": "Sergio Andrés Ipuz Saavedra",
        "course": "Developing the Artificial Intelligence of Things (CSM13111)",
        "university": "University of Helsinki",
        "architecture": "MLP (14 -> 8 -> 1)",
        "quantization": "Symmetric INT8 Post-Training Quantization",
        "test_accuracy": float(test_acc),
        "flash_bytes": total_weights_bytes,
        "sram_bytes": hidden_dim * 1 + 14 * 1
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported metadata: {meta_path}")

if __name__ == "__main__":
    train_and_quantize()
