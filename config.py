"""
config.py — QPKI-SmartHome Configuration
All hyperparameters, paths, and settings from the paper.

Paper: "QPKI-SmartHome: Post-Quantum Public Key Infrastructure with
        Lightweight Lattice-Based Authentication for Cloud-Tethered IoT
        Smart Home Devices"
Journal: MDPI Sensors, 2026
"""

import os

# =============================================================================
# Section 6.1 — Evaluation Platform
# =============================================================================
HARDWARE = {
    "mcu": "STM32F429ZI (ARM Cortex-M4 @ 168 MHz)",
    "ram": 262144,           # 256 KB in bytes
    "flash": 2097152,        # 2 MB in bytes
    "edge_gateway": "Raspberry Pi 4 (Cortex-A72 @ 1.5 GHz)",
    "wireless_standard": "IEEE 802.15.4",
    "wireless_bandwidth_bps": 250000,  # 250 kbps
}

# =============================================================================
# Section 4.1 — ML-KEM Parameter Sets (NIST FIPS 203)
# =============================================================================
ML_KEM = {
    "ML-KEM-512": {
        "security_level": 1,
        "pk_bytes": 800,
        "sk_bytes": 1632,
        "ct_bytes": 768,
        "energy_mJ": 3.05,
        "description": "Low battery / low threat",
    },
    "ML-KEM-768": {
        "security_level": 3,
        "pk_bytes": 1184,
        "sk_bytes": 2400,
        "ct_bytes": 1088,
        "energy_mJ": 4.86,
        "description": "Normal operation",
    },
    "ML-KEM-1024": {
        "security_level": 5,
        "pk_bytes": 1568,
        "sk_bytes": 3168,
        "ct_bytes": 1568,
        "energy_mJ": 7.00,
        "description": "High threat level",
    },
}

# =============================================================================
# Section 4.1 — ML-DSA Parameter Sets (NIST FIPS 204)
# =============================================================================
ML_DSA = {
    "ML-DSA-44": {
        "security_level": 2,
        "pk_bytes": 1312,
        "sk_bytes": 2528,
        "sig_bytes": 2420,
        "role": "device_certificate",
        "description": "MCU-class device (leaf)",
    },
    "ML-DSA-65": {
        "security_level": 3,
        "pk_bytes": 1952,
        "sk_bytes": 4000,
        "sig_bytes": 3293,
        "role": "intermediate_ca",
        "description": "Intermediate CA",
    },
    "ML-DSA-87": {
        "security_level": 5,
        "pk_bytes": 2592,
        "sk_bytes": 4864,
        "sig_bytes": 4595,
        "role": "root_ca",
        "description": "Root CA",
    },
}

# =============================================================================
# Section 4.2 — Certificate Compression (Table 3 values)
# =============================================================================
CERTIFICATE_COMPRESSION = {
    # ML-DSA-65 intermediate CA certificate
    "full_der_bytes": 5627,
    "after_delta_strip_bytes": 5261,   # after removing chain-shared fields
    "after_zstd_bytes": 1805,          # after Zstandard entropy coding
    "compression_ratio": 0.61,         # 61% size reduction
    # Full chain for 256 KB RAM constraint
    "chain_compressed_bytes": 4200,    # ≈4.2 KB total chain
    "ecdsa_p256_cert_bytes": 256,      # classical baseline
}

# =============================================================================
# Section 4.3 — Energy-Aware Handshake Scheduler
# =============================================================================
SCHEDULER = {
    "battery_thresholds": {
        "low": 0.20,      # below 20% → ML-KEM-512
        "medium": 0.50,   # 20–50% → ML-KEM-768
        "high": 1.00,     # above 50% → ML-KEM-1024 when threat elevated
    },
    "threat_thresholds": {
        "low": 0.30,
        "medium": 0.70,
        "high": 1.00,
    },
    "weighted_avg_energy_mJ": 4.749,
    "energy_weights": {
        "ML-KEM-512": 0.333,
        "ML-KEM-768": 0.334,
        "ML-KEM-1024": 0.333,
    },
}

# =============================================================================
# Section 6.2 — Performance Results (Table 4)
# =============================================================================
PERFORMANCE_RESULTS = {
    "computation_latency_ms": 38,
    "end_to_end_latency_ms": 545.6,
    "energy_per_handshake_mJ": 4.749,
    "peak_ram_bytes": 51200,          # 50 KB peak RAM
    "auth_success_rate": 0.997,       # 99.7%
    "bypass_probability": 0.003,
    "cert_compression_ratio": 0.61,
    "throughput_handshakes_per_s": 26.3,
}

# =============================================================================
# Section 6.3 — Baseline Comparisons (Table 5)
# =============================================================================
BASELINES = {
    "PQ-TLS-IoT":    {"energy_mJ": 18.7, "latency_ms": 142, "cert_kb": 10.8},
    "Kyber-Raw":     {"energy_mJ": 12.3, "latency_ms":  89, "cert_kb":  6.2},
    "ECDSA-TLS":     {"energy_mJ":  2.1, "latency_ms":  31, "cert_kb":  0.5},
    "LWE-Auth":      {"energy_mJ":  9.4, "latency_ms":  76, "cert_kb":  4.1},
    "HybridPQ":      {"energy_mJ": 14.2, "latency_ms": 118, "cert_kb":  8.9},
    "NTRU-PKI":      {"energy_mJ": 11.8, "latency_ms":  95, "cert_kb":  5.7},
    "QPKI-SmartHome":{"energy_mJ":  4.749,"latency_ms": 38, "cert_kb":  4.2},
}

# =============================================================================
# Datasets — Section 6.1
# =============================================================================
DATASETS = {
    "CIC-IoT-2023": {
        "url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "download_url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "local_dir": "data/CIC-IoT-2023",
        "description": "Primary dataset for traffic anomaly scoring / threat classification",
        "num_classes": 34,
        "total_samples": 3048,
        "features": 46,
    },
    "TON_IoT": {
        "url": "https://research.unsw.edu.au/projects/toniot-datasets",
        "download_url": "https://research.unsw.edu.au/projects/toniot-datasets",
        "local_dir": "data/TON_IoT",
        "description": "Cross-dataset robustness validation",
    },
    "Bot-IoT": {
        "url": "https://research.unsw.edu.au/projects/bot-iot-dataset",
        "download_url": "https://research.unsw.edu.au/projects/bot-iot-dataset",
        "local_dir": "data/Bot-IoT",
        "description": "Cross-dataset robustness validation",
    },
}

# =============================================================================
# Threat Classifier Hyperparameters — Section 4.3
# =============================================================================
CLASSIFIER = {
    # Random Forest used for threat-level scoring (Raspberry Pi 4 edge gateway)
    "model_type": "RandomForest",
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42,
    # Train / Val / Test split
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Feature selection
    "n_features_selected": 20,
    "feature_selection_method": "mutual_info",
}

# =============================================================================
# DNN Anomaly Detector Hyperparameters (secondary model, edge gateway)
# =============================================================================
DNN = {
    "input_dim": 20,          # after feature selection
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "activation": "ReLU",
    "output_dim": 2,          # binary: benign / attack
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "T_max": 50,
    "early_stopping_patience": 10,
}

# =============================================================================
# File Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for _d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "last_model.pt")
RF_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "rf_classifier.pkl")
RESULTS_CSV     = os.path.join(RESULTS_DIR, "evaluation_results.csv")
CONFUSION_MATRIX_PNG = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PNG        = os.path.join(RESULTS_DIR, "roc_curve.png")
