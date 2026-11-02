"""
dataset.py — QPKI-SmartHome Dataset Loading and Preprocessing
CIC-IoT-2023, TON_IoT, Bot-IoT datasets.

Section 6.1: Datasets used for threat-level classification on the
Raspberry Pi 4 edge gateway (energy-aware scheduler replay evaluation).

Dataset: CIC-IoT-2023
Download: https://www.unb.ca/cic/datasets/iotdataset-2023.html
Paper: Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for
       Large-Scale Attacks in IoT Environments", Sensors 2023.
"""

import os
import sys
import urllib.request
import zipfile
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATASETS, CLASSIFIER, DATA_DIR


# =============================================================================
# Section 6.1 — Dataset Download
# =============================================================================
CIC_IOT_FILES = [
    # CIC-IoT-2023 CSV files — download from the UNB ISCX portal
    # Direct CSVs are available after registration at the URL below.
    # Mirror / Kaggle version: https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
    "https://www.kaggle.com/api/v1/datasets/download/madhavmalhotra/unb-cic-iot-dataset",
]


def download_cic_iot_2023(dest_dir: str = None) -> str:
    """
    Download CIC-IoT-2023 dataset.
    Requires Kaggle API credentials in ~/.kaggle/kaggle.json.
    Alternatively, place CSV files manually in data/CIC-IoT-2023/.

    Returns path to the local dataset directory.
    """
    if dest_dir is None:
        dest_dir = os.path.join(DATA_DIR, "CIC-IoT-2023")
    os.makedirs(dest_dir, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(dest_dir) if f.endswith(".csv")]
    if existing:
        print(f"[dataset] Found {len(existing)} CSV(s) in {dest_dir}. Skipping download.")
        return dest_dir

    try:
        import kaggle  # pip install kaggle
        print("[dataset] Downloading CIC-IoT-2023 via Kaggle API …")
        kaggle.api.dataset_download_files(
            "madhavmalhotra/unb-cic-iot-dataset",
            path=dest_dir,
            unzip=True,
        )
        print(f"[dataset] Download complete → {dest_dir}")
    except ImportError:
        print(
            "[dataset] kaggle package not found.\n"
            "Install: pip install kaggle\n"
            "Then place your kaggle.json at ~/.kaggle/kaggle.json\n"
            f"Or manually download from https://www.unb.ca/cic/datasets/iotdataset-2023.html\n"
            f"and place CSVs in: {dest_dir}"
        )
        sys.exit(1)
    except Exception as e:
        print(f"[dataset] Download failed: {e}")
        print(f"Please download manually and place CSVs in: {dest_dir}")
        sys.exit(1)

    return dest_dir


# =============================================================================
# Section 6.1 — Feature Extraction (46 network flow features)
# =============================================================================

# 46 raw features in CIC-IoT-2023 (network flow statistics)
FLOW_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type",
    "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
]

LABEL_COLUMN = "label"


def load_cic_iot_2023(data_dir: str = None, sample_frac: float = 1.0):
    """
    Load CIC-IoT-2023 CSV files into a single DataFrame.
    Applies basic cleaning and type coercion.

    Args:
        data_dir: path to directory containing CSV files
        sample_frac: fraction of each CSV to sample (0–1), for quick testing

    Returns:
        (X, y) — numpy arrays, X shape (N, 46), y shape (N,) int labels
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "CIC-IoT-2023")

    csv_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.endswith(".csv")
    ])
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Run: python dataset.py --download"
        )

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[dataset] Loaded {len(df):,} rows from {len(csv_files)} CSV(s)")

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Identify label column (case-insensitive)
    label_cols = [c for c in df.columns if c.lower() == "label"]
    if not label_cols:
        raise ValueError(f"No 'label' column found. Available: {df.columns.tolist()}")
    label_col = label_cols[0]

    # Select feature columns that exist
    available = [c for c in FLOW_FEATURES if c in df.columns]
    missing = [c for c in FLOW_FEATURES if c not in df.columns]
    if missing:
        print(f"[dataset] Warning: {len(missing)} features missing, using {len(available)} available.")

    X_df = df[available].copy()
    y_raw = df[label_col].copy()

    # Coerce to numeric; drop rows with NaN
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    mask = X_df.notna().all(axis=1)
    X_df = X_df[mask]
    y_raw = y_raw[mask]

    # Replace inf
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_df.fillna(0, inplace=True)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    print(f"[dataset] Classes ({len(le.classes_)}): {le.classes_}")
    print(f"[dataset] Final shape: X={X_df.shape}, y={y.shape}")
    return X_df.values.astype(np.float32), y.astype(np.int64), le


# =============================================================================
# Section 4.3 — Feature Selection (mutual information, top 20)
# =============================================================================

def select_features(X_train, y_train, n_features: int = 20):
    """
    Select top-n features by mutual information with label.
    Section 4.3: 20 features used by the threat-level classifier.

    Returns selected indices and importance scores.
    """
    mi = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42)
    idx = np.argsort(mi)[::-1][:n_features]
    print(f"[dataset] Top {n_features} feature indices: {idx}")
    return idx, mi


# =============================================================================
# Section 6.1 — Train / Val / Test Split (70 / 15 / 15)
# =============================================================================

def split_data(X, y, train_ratio=0.70, val_ratio=0.15, random_state=42):
    """
    Stratified 70/15/15 split.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    test_size = 1.0 - train_ratio
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_ratio / test_size
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - relative_val, stratify=y_tmp, random_state=random_state
    )
    print(f"[dataset] Split → train={len(X_tr):,} val={len(X_val):,} test={len(X_te):,}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# =============================================================================
# PyTorch Dataset / DataLoader
# =============================================================================

class IoTFlowDataset(Dataset):
    """
    PyTorch Dataset wrapping scaled IoT network flow features.
    Used by the DNN anomaly detector on the edge gateway.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    batch_size: int = 256,
    num_workers: int = 2,
):
    """Build train / val / test DataLoaders."""
    train_ds = IoTFlowDataset(X_train, y_train)
    val_ds   = IoTFlowDataset(X_val,   y_val)
    test_ds  = IoTFlowDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# =============================================================================
# download_data.py shim — callable directly for data/ setup
# =============================================================================

def prepare_dataset(download: bool = False, sample_frac: float = 1.0):
    """
    Full pipeline: optionally download → load → split → scale → select features.
    Returns dict with all splits and scaler.
    """
    data_dir = os.path.join(DATA_DIR, "CIC-IoT-2023")
    if download:
        download_cic_iot_2023(data_dir)

    X, y, le = load_cic_iot_2023(data_dir, sample_frac=sample_frac)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(
        X, y,
        train_ratio=CLASSIFIER["train_ratio"],
        val_ratio=CLASSIFIER["val_ratio"],
    )
    scaler = fit_scaler(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    # Feature selection on training set
    feat_idx, mi_scores = select_features(X_tr_s, y_tr, CLASSIFIER["n_features_selected"])
    X_tr_s  = X_tr_s[:, feat_idx]
    X_val_s = X_val_s[:, feat_idx]
    X_te_s  = X_te_s[:, feat_idx]

    return {
        "X_train": X_tr_s,  "y_train": y_tr,
        "X_val":   X_val_s, "y_val":   y_val,
        "X_test":  X_te_s,  "y_test":  y_te,
        "scaler":  scaler,
        "feat_idx": feat_idx,
        "label_encoder": le,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QPKI-SmartHome dataset utility")
    parser.add_argument("--download", action="store_true", help="Download CIC-IoT-2023")
    parser.add_argument("--sample-frac", type=float, default=0.1,
                        help="Fraction to load for quick testing (default 0.1)")
    args = parser.parse_args()
    data = prepare_dataset(download=args.download, sample_frac=args.sample_frac)
    print("[dataset] Ready. X_train shape:", data["X_train"].shape)
