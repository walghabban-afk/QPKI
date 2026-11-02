"""
inference.py — QPKI-SmartHome Inference

Given a single network flow feature vector (or a CSV row), runs the
ThreatClassifier to produce a threat score, then invokes the
EnergyAwareScheduler to select the appropriate ML-KEM parameter set
for the next handshake.

Also simulates the full QPKI handshake and reports energy / latency.

Usage:
    python inference.py --battery 0.45 --csv data/sample_flow.csv
    python inference.py --random-flow  # generate random flow for demo
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import BEST_CHECKPOINT, RESULTS_DIR, ML_KEM
from model import ThreatClassifier, EnergyAwareScheduler, QPKIHandshakeSimulator
from dataset import FLOW_FEATURES


# =============================================================================
# Load model
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run: python train.py  first."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ThreatClassifier(
        input_dim=ckpt["input_dim"],
        n_classes=ckpt["n_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# =============================================================================
# Pre-process a single flow record
# =============================================================================

def preprocess_flow(raw: np.ndarray, scaler=None, feat_idx=None) -> torch.Tensor:
    """
    Apply the same scaling and feature selection used during training.

    Args:
        raw:      1D numpy array of raw flow features (46 values)
        scaler:   sklearn StandardScaler fitted on training set
        feat_idx: array of selected feature indices (length 20)

    Returns:
        Tensor of shape (1, n_selected_features)
    """
    x = raw.copy().astype(np.float32).reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)
    if feat_idx is not None:
        x = x[:, feat_idx]
    return torch.from_numpy(x)


# =============================================================================
# Section 4.3 — Full Inference Pipeline
# =============================================================================

def run_inference(
    model: ThreatClassifier,
    feat_tensor: torch.Tensor,
    battery_level: float,
    device: torch.device,
    label_encoder=None,
) -> dict:
    """
    Run the full QPKI-SmartHome inference pipeline:
      1. ThreatClassifier → class probabilities + threat score
      2. EnergyAwareScheduler → ML-KEM parameter selection
      3. QPKIHandshakeSimulator → predicted energy, latency, RAM

    Returns result dict.
    """
    feat_tensor = feat_tensor.to(device)

    with torch.no_grad():
        logits = model(feat_tensor)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Threat score = P(attack) in binary case, else 1 - P(benign)
    n_classes = probs.shape[0]
    threat_score = float(probs[1]) if n_classes == 2 else float(1.0 - probs[0])
    pred_class_idx = int(np.argmax(probs))

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        pred_label = label_encoder.classes_[pred_class_idx]
    else:
        pred_label = str(pred_class_idx)

    # Scheduler decision
    sched = EnergyAwareScheduler()
    kem_choice = sched.select_kem(battery=battery_level, threat=threat_score)
    energy_mJ  = sched.get_energy(kem_choice)

    # Handshake simulation
    sim = QPKIHandshakeSimulator()
    handshake = sim.simulate(kem_choice, "ML-DSA-44")

    result = {
        "predicted_class": pred_label,
        "threat_score": round(threat_score, 4),
        "battery_level": battery_level,
        "selected_kem": kem_choice,
        "security_level": ML_KEM[kem_choice]["security_level"],
        "energy_mJ": energy_mJ,
        "computation_latency_ms": handshake["computation_latency_ms"],
        "end_to_end_latency_ms": handshake["end_to_end_latency_ms"],
        "peak_ram_bytes": handshake["peak_ram_bytes"],
        "class_probabilities": {
            str(i): round(float(p), 4) for i, p in enumerate(probs)
        },
    }
    return result


# =============================================================================
# Visualization
# =============================================================================

def visualize_result(result: dict, save_path: str):
    """
    Save a summary figure showing threat score, KEM selection, and energy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("QPKI-SmartHome Inference Summary", fontsize=13, fontweight="bold")

    # Panel 1: threat score gauge
    ax = axes[0]
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", lw=2)
    score = result["threat_score"]
    needle_angle = np.pi * (1 - score)
    ax.annotate("", xy=(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Threat Score: {score:.3f}", fontsize=11)
    ax.text(0, -0.15, f"Battery: {result['battery_level']:.0%}", ha="center", fontsize=10)

    # Panel 2: KEM selection bar
    ax = axes[1]
    kem_labels = list(ML_KEM.keys())
    kem_energies = [ML_KEM[k]["energy_mJ"] for k in kem_labels]
    colors = ["#2ecc71" if k == result["selected_kem"] else "#bdc3c7" for k in kem_labels]
    bars = ax.bar(kem_labels, kem_energies, color=colors, edgecolor="white")
    ax.set_ylabel("Energy (mJ)", fontsize=10)
    ax.set_title(f"Selected: {result['selected_kem']}", fontsize=11)
    for bar, e in zip(bars, kem_energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{e:.2f}", ha="center", fontsize=9)

    # Panel 3: latency breakdown
    ax = axes[2]
    labels = ["Computation", "Transmission"]
    comp = result["computation_latency_ms"]
    trans = result["end_to_end_latency_ms"] - comp
    values = [comp, trans]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=["#3498db", "#e67e22"], startangle=90
    )
    ax.set_title(f"E2E Latency: {result['end_to_end_latency_ms']:.1f} ms", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[inference] Result figure saved → {save_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="QPKI-SmartHome inference")
    p.add_argument("--checkpoint", default=BEST_CHECKPOINT)
    p.add_argument("--csv",        default=None,  help="CSV file with one flow row")
    p.add_argument("--battery",    type=float, default=0.45, help="Battery level 0–1")
    p.add_argument("--random-flow", action="store_true", help="Use random feature vector")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    model, ckpt = load_model(args.checkpoint, device)
    feat_idx = ckpt.get("feat_idx")

    # Load or generate input
    if args.csv is not None:
        import pandas as pd
        df = pd.read_csv(args.csv)
        available = [c for c in FLOW_FEATURES if c in df.columns]
        raw = df[available].iloc[0].values.astype(np.float32)
        # Pad to expected size if features differ
        if len(raw) < len(FLOW_FEATURES):
            pad = np.zeros(len(FLOW_FEATURES) - len(raw), dtype=np.float32)
            raw = np.concatenate([raw, pad])
    else:
        print("[inference] No CSV provided — using random flow vector.")
        rng = np.random.default_rng(42)
        raw = rng.uniform(0, 1, size=len(FLOW_FEATURES)).astype(np.float32)

    feat_tensor = preprocess_flow(raw, feat_idx=feat_idx)

    result = run_inference(model, feat_tensor, args.battery, device)

    print("\n[inference] Result:")
    for k, v in result.items():
        if k != "class_probabilities":
            print(f"  {k:<35s} {v}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = os.path.join(RESULTS_DIR, "inference_result.png")
    visualize_result(result, fig_path)

    json_path = os.path.join(RESULTS_DIR, "inference_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[inference] JSON saved → {json_path}")


if __name__ == "__main__":
    main()
