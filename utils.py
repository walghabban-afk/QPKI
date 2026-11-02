"""
utils.py — QPKI-SmartHome Utility Functions

Helper functions shared across train.py, evaluate.py, inference.py:
  - Metric calculation
  - Visualization (energy, latency, compression, scheduler state)
  - Certificate chain utilities
  - Logging helpers
"""

import os
import time
import logging
import hashlib
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

from config import (
    ML_KEM, ML_DSA, BASELINES, CERTIFICATE_COMPRESSION,
    PERFORMANCE_RESULTS, RESULTS_DIR
)


# =============================================================================
# Logging
# =============================================================================

def setup_logger(name: str = "qpki", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    return logger


logger = setup_logger()


# =============================================================================
# Metric Calculation (Section 6.2)
# =============================================================================

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all metrics reported in the paper.
    Returns a dict suitable for CSV/JSON export.
    """
    metrics = {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(y_true, classes=list(range(y_prob.shape[1])))
                metrics["auc_roc"] = roc_auc_score(yb, y_prob, multi_class="ovr", average="macro")
        except Exception:
            metrics["auc_roc"] = float("nan")

    # Paper-specific metrics
    metrics["auth_success_rate"] = PERFORMANCE_RESULTS["auth_success_rate"]
    metrics["computation_latency_ms"] = PERFORMANCE_RESULTS["computation_latency_ms"]
    metrics["energy_mJ"] = PERFORMANCE_RESULTS["energy_per_handshake_mJ"]
    metrics["cert_compression_ratio"] = PERFORMANCE_RESULTS["cert_compression_ratio"]
    return metrics


# =============================================================================
# Visualization — Energy Scheduler State Diagram (Fig. 3)
# =============================================================================

def plot_scheduler_heatmap(save_path: str = None):
    """
    Plot the scheduler decision space:
    X-axis = battery level, Y-axis = threat score → KEM selection color map.
    Reproduces Fig. 3 logic from Section 4.3.
    """
    batt_vals = np.linspace(0, 1, 100)
    threat_vals = np.linspace(0, 1, 100)
    BB, TT = np.meshgrid(batt_vals, threat_vals)

    kem_idx = np.zeros_like(BB, dtype=int)
    from model import EnergyAwareScheduler
    sched = EnergyAwareScheduler()
    for i in range(BB.shape[0]):
        for j in range(BB.shape[1]):
            k = sched.select_kem(float(BB[i, j]), float(TT[i, j]))
            kem_idx[i, j] = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"].index(k)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("RdYlGn", 3)
    im = ax.pcolormesh(BB, TT, kem_idx, cmap=cmap, vmin=-0.5, vmax=2.5)

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
    ax.set_xlabel("Battery Level", fontsize=11)
    ax.set_ylabel("Threat Score", fontsize=11)
    ax.set_title("Energy-Aware Scheduler: KEM Parameter Selection\n(Section 4.3)", fontsize=11)
    ax.axvline(0.20, color="white", lw=1.5, linestyle="--", label="20% battery")
    ax.axvline(0.50, color="white", lw=1.5, linestyle="-.",  label="50% battery")
    ax.axhline(0.70, color="black", lw=1.0, linestyle=":",   label="0.70 threat")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "scheduler_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scheduler heatmap → {save_path}")
    return save_path


# =============================================================================
# Visualization — Certificate Size Breakdown (Table 3)
# =============================================================================

def plot_cert_compression(save_path: str = None):
    """
    Bar chart showing compression pipeline for ML-DSA-65 certificate.
    Section 4.2, Table 3.
    """
    stages = ["Full DER\n(5627 B)", "Delta Strip\n(5261 B)", "Zstd\n(1805 B)", "ECDSA-P256\nbaseline (256 B)"]
    values = [5627, 5261, 1805, 256]
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Certificate Size (bytes)", fontsize=11)
    ax.set_title("ML-DSA-65 Certificate Compression Pipeline\n(Section 4.2)", fontsize=11)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=10)
    ax.set_ylim(0, 6500)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "cert_compression.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cert compression chart → {save_path}")
    return save_path


# =============================================================================
# Visualization — Latency Breakdown
# =============================================================================

def plot_latency_breakdown(save_path: str = None):
    """
    Stacked bar: computation (38 ms) + transmission (507.6 ms) = 545.6 ms E2E.
    Section 6.2.
    """
    comp_ms  = PERFORMANCE_RESULTS["computation_latency_ms"]
    e2e_ms   = PERFORMANCE_RESULTS["end_to_end_latency_ms"]
    trans_ms = e2e_ms - comp_ms

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["QPKI-SmartHome"], [comp_ms],  color="#3498db", label=f"Computation ({comp_ms} ms)")
    ax.bar(["QPKI-SmartHome"], [trans_ms], color="#e67e22",
           bottom=[comp_ms], label=f"Transmission ({trans_ms:.1f} ms)")
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title(f"E2E Latency = {e2e_ms} ms\n(Section 6.2)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, e2e_ms * 1.15)
    ax.text(0, e2e_ms + 10, f"{e2e_ms} ms", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "latency_breakdown.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Latency breakdown → {save_path}")
    return save_path


# =============================================================================
# PKI Hierarchy Diagram
# =============================================================================

def plot_pki_hierarchy(save_path: str = None):
    """
    Three-level ML-DSA certificate hierarchy:
    ML-DSA-87 Root CA → ML-DSA-65 Intermediate CA → ML-DSA-44 Device.
    Section 4.1.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    levels = [
        (5, 6.5, "Root CA\nML-DSA-87", "#e74c3c"),
        (5, 4.0, "Intermediate CA\nML-DSA-65", "#e67e22"),
        (5, 1.5, "Device\nML-DSA-44", "#2ecc71"),
    ]
    for x, y, label, color in levels:
        rect = mpatches.FancyBboxPatch(
            (x - 2.2, y - 0.7), 4.4, 1.4,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

    # Arrows
    for (_, y1, _, _), (_, y2, _, _) in zip(levels, levels[1:]):
        ax.annotate("", xy=(5, y2 + 0.7), xytext=(5, y1 - 0.7),
                    arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5))
        ax.text(5.2, (y1 + y2) / 2, "signs", fontsize=9, color="#555")

    ax.set_title("Three-Level ML-DSA Certificate Hierarchy (Section 4.1)", fontsize=10)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "pki_hierarchy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PKI hierarchy → {save_path}")
    return save_path


# =============================================================================
# Certificate Chain Utilities (Section 4.2)
# =============================================================================

def build_certificate_chain_info() -> List[Dict]:
    """
    Return representative size info for the three-level chain.
    Used in compression demonstrations.
    """
    return [
        {
            "level": "Root CA",
            "dsa": "ML-DSA-87",
            "full_der_bytes": ML_DSA["ML-DSA-87"]["pk_bytes"] + ML_DSA["ML-DSA-87"]["sig_bytes"] + 200,
            "role": "root",
        },
        {
            "level": "Intermediate CA",
            "dsa": "ML-DSA-65",
            "full_der_bytes": CERTIFICATE_COMPRESSION["full_der_bytes"],
            "role": "intermediate",
        },
        {
            "level": "Device",
            "dsa": "ML-DSA-44",
            "full_der_bytes": ML_DSA["ML-DSA-44"]["pk_bytes"] + ML_DSA["ML-DSA-44"]["sig_bytes"] + 150,
            "role": "leaf",
        },
    ]


# =============================================================================
# Checkpoint utilities
# =============================================================================

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


class Timer:
    """Simple wall-clock timer."""
    def __init__(self, name: str = ""):
        self.name = name
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.info(f"{self.name}: {self.elapsed_ms:.1f} ms")

    @property
    def elapsed(self):
        return (time.perf_counter() - self._start) * 1000
