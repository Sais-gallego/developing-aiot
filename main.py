"""
Developing the Artificial Intelligence of Things (AIoT)
Main Simulation Entrypoint & Demonstration
University of Helsinki - Coursework & Simulation Project
Author: Sergio Andrés Ipuz Saavedra (@Sais-gallego)
"""

import json
from src.pipeline import AioTPipeline


def main():
    print("=" * 70)
    print("  DEVELOPING THE ARTIFICIAL INTELLIGENCE OF THINGS (AIoT)")
    print("  University of Helsinki - Simulation Project")
    print("  Author: Sergio Andrés Ipuz Saavedra (@Sais-gallego)")
    print("=" * 70)

    print("\n[Step 1] Initializing Simulated Edge Node & Calibration Data...")
    pipeline = AioTPipeline(node_id="helsinki-edge-node-01")
    train_stats = pipeline.train_and_calibrate(num_samples_per_class=120)
    print(f" -> TinyML Model Trained. Final BCE Loss: {train_stats['final_loss']:.5f}")

    print("\n[Step 2] Microcontroller Quantization & Memory Budget Profiling:")
    footprint = pipeline.model.estimate_hardware_footprint()
    print(f" -> Total Model Parameters:       {int(footprint['parameters_count'])}")
    print(f" -> Float32 Reference Size:       {footprint['fp32_model_size_bytes']:.0f} bytes")
    print(f" -> Quantized INT8 Size:          {footprint['int8_model_size_bytes']:.0f} bytes")
    print(f" -> Model Compression Ratio:      {footprint['compression_ratio']:.2f}x")
    print(f" -> SRAM Activation Buffer:       {footprint['sram_buffer_bytes']:.0f} bytes (< 1 KB SRAM required!)")

    print("\n[Step 3] Running 25-Window Continuous Simulation with Anomaly Injections...")
    results = pipeline.run_simulation(total_windows=25, anomaly_frequency=5)

    print(f" -> Overall Edge Classification Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f" -> Bandwidth Reduction vs Cloud Stream:   {results['bandwidth_saved_percentage']:.2f}%")

    print("\n[Step 4] Recent Telemetry / Anomaly Alerts Log:")
    for win in results["simulation_details"][-8:]:
        status_icon = "🚨 CRITICAL" if win["predicted_anomaly"] else "✅ NORMAL  "
        print(
            f"   Window #{win['window']:02d} | Status: {status_icon} | "
            f"Conf: {win['confidence'] * 100:5.1f}% | "
            f"Latency: {win['latency_us']:5.1f} us | "
            f"Energy: {win['energy_uj']:5.2f} uJ | "
            f"MQTT: {win['mqtt_topic']}"
        )

    print("\n" + "=" * 70)
    print("  Simulation completed successfully! Project ready for submission.")
    print("=" * 70)


if __name__ == "__main__":
    main()
