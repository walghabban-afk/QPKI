# QPKI-SmartHome

**QPKI-SmartHome: Post-Quantum Public Key Infrastructure with Lightweight Lattice-Based Authentication for Cloud-Tethered IoT Smart Home Devices**

**Journal:** MDPI Sensors, 2026

---

## Abstract

QPKI-SmartHome is a unified post-quantum PKI framework co-designed around NIST FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA) for cloud-tethered smart home IoT networks. It introduces delta-encoding certificate compression (61% size reduction), an energy-aware handshake scheduler (4.749 mJ per handshake, 74% energy reduction vs PQ-TLS-IoT), and an automated cloud PKI orchestrator. Evaluated on ARM Cortex-M4 (STM32F429ZI) with CIC-IoT-2023, TON_IoT, and Bot-IoT datasets.

---

## Key Results (Table 4)

| Metric | Value |
|--------|-------|
| Computation Latency | 38 ms |
| End-to-End Latency | 545.6 ms |
| Energy per Handshake | 4.749 mJ |
| Certificate Compression | 61% |
| Peak RAM | 50 KB |
| Auth Success Rate | 99.7% |
| Energy Reduction vs PQ-TLS-IoT | 74% |

---

## Architecture

**Two-track evaluation system:**

1. **STM32F429ZI MCU** (ARM Cortex-M4, 256 KB RAM): Runs embedded C implementation of ML-KEM key encapsulation and ML-DSA certificate verification. Measures computation latency and energy directly on hardware.

2. **Raspberry Pi 4 edge gateway**: Runs the Python ThreatClassifier (DNN / Random Forest) on network flow features from CIC-IoT-2023. The classifier output drives the energy-aware scheduler for ML-KEM parameter selection.

**Three PKI layers:**
- ML-DSA-87 Root CA (cloud)
- ML-DSA-65 Intermediate CA (cloud)
- ML-DSA-44 Device (MCU leaf)

---

## Dataset

**Dataset:** CIC-IoT-2023  
**Download:** https://www.unb.ca/cic/datasets/iotdataset-2023.html  
**Kaggle mirror:** https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset

Place downloaded CSV files in:
```
data/
  CIC-IoT-2023/
    *.csv
```

Or run the download script:
```bash
python data/download_data.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

For post-quantum primitives (ML-KEM / ML-DSA simulation):
```bash
pip install liboqs-python
```

---

## How to Train

```bash
# Full dataset
python train.py

# Quick test with 10% data
python train.py --sample-frac 0.1 --epochs 5

# Also train Random Forest baseline
python train.py --rf
```

Training logs appear in `logs/` and can be viewed with TensorBoard:
```bash
tensorboard --logdir logs/
```

---

## How to Evaluate

```bash
python evaluate.py
```

Outputs:
- `results/evaluation_results.csv` — metrics table
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/energy_comparison.png`

---

## How to Run Inference

```bash
# With random flow (demo)
python inference.py --random-flow --battery 0.45

# With a CSV file
python inference.py --csv data/sample_flow.csv --battery 0.35
```

---

## Folder Structure

```
QPKI_SmartHome_Implementation/
├── figures/                    # Extracted paper figures (HD)
│   ├── fig01_system_architecture.png
│   ├── fig02_certificate_compression.png
│   ├── fig03_energy_scheduler.png
│   └── ...
├── data/                       # Dataset (place CIC-IoT-2023 CSVs here)
│   └── download_data.py
├── checkpoints/                # Saved model weights
├── results/                    # Evaluation outputs
├── notebooks/
│   └── demo.ipynb              # Step-by-step walkthrough
├── config.py                   # All hyperparameters from the paper
├── dataset.py                  # CIC-IoT-2023 loading & preprocessing
├── model.py                    # ThreatClassifier, EnergyAwareScheduler, Simulator
├── train.py                    # Training loop (DNN + Random Forest)
├── evaluate.py                 # Metrics, confusion matrix, ROC, baselines
├── inference.py                # Single-flow inference + visualization
├── utils.py                    # Helpers, plots, certificate chain utilities
└── requirements.txt
```

---

## Reproducing Paper Results

The paper reports two evaluation tracks:

**Track 1 (Hardware):** Computation latency (38 ms), energy (4.749 mJ), peak RAM (50 KB) — measured on STM32F429ZI hardware running the embedded C implementation. Not reproducible in Python; see the paper's embedded C code description in Section 5.

**Track 2 (Software / Dataset):** Authentication success rate (99.7%), cross-dataset robustness — reproducible by running `train.py` and `evaluate.py` on CIC-IoT-2023.

---

## License

Implementation code released under MIT License.  
Paper figures are from the MDPI Sensors publication (CC BY 4.0).

---

## Citation

If you use this code, please cite:

```bibtex
@article{qpki_smarthome_2026,
  title   = {QPKI-SmartHome: Post-Quantum Public Key Infrastructure with
             Lightweight Lattice-Based Authentication for Cloud-Tethered
             IoT Smart Home Devices},
  journal = {MDPI Sensors},
  year    = {2026},
}
```
