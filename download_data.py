"""
download_data.py — Download CIC-IoT-2023 dataset

Usage:
    python data/download_data.py

Requires: pip install kaggle
Then: place kaggle.json at ~/.kaggle/kaggle.json
      (download from https://www.kaggle.com/settings → API → Create New Token)

Dataset: CIC-IoT-2023
Source:  https://www.unb.ca/cic/datasets/iotdataset-2023.html
Kaggle:  https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import download_cic_iot_2023

if __name__ == "__main__":
    dest = os.path.join(os.path.dirname(__file__), "CIC-IoT-2023")
    download_cic_iot_2023(dest)
    print(f"Dataset saved to: {dest}")
