"""
model.py — QPKI-SmartHome Model Definitions

Two models are implemented:

1. ThreatClassifier (DNN) — edge gateway (Raspberry Pi 4)
   Section 4.3: IoT traffic anomaly scoring for energy-aware scheduler.
   Input: 20 selected network flow features.
   Architecture: MLP [128 → 64 → 32 → n_classes] with BatchNorm + Dropout.

2. EnergyAwareScheduler — rule-based + classifier output
   Section 4.3: maps (battery_state, threat_score) → ML-KEM parameter set.

3. QPKIHandshakeSimulator — simulates the MCU-side handshake
   Section 6.2: computes energy, latency, RAM given chosen KEM params.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ML_KEM, ML_DSA, SCHEDULER, PERFORMANCE_RESULTS, DNN, CERTIFICATE_COMPRESSION
)


# =============================================================================
# Section 4.3 — DNN Threat Classifier (edge gateway)
# =============================================================================

class ResidualBlock(nn.Module):
    """One residual block: Linear → BN → ReLU → Dropout → Linear → BN → +skip."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ThreatClassifier(nn.Module):
    """
    DNN anomaly detector / threat classifier.

    Section 4.3: deployed on Raspberry Pi 4 edge gateway.
    Classifies network flows from CIC-IoT-2023 into benign / attack classes.
    Output is a threat score in [0, 1] used by the energy-aware scheduler.

    Architecture (from paper Table 2 hyperparameters):
        Input(20) → Linear(128) → BN → ReLU → Dropout(0.3)
                  → Linear(64)  → BN → ReLU → Dropout(0.3)
                  → Linear(32)  → BN → ReLU → Dropout(0.3)
                  → Linear(n_classes)
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: list = None,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DNN["hidden_dims"]  # [128, 64, 32]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, n_classes)."""
        return self.net(x)

    def predict_threat_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar threat score in [0, 1].
        P(attack) — probability of the traffic being adversarial.
        Used by EnergyAwareScheduler.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            # Binary case: class 1 = attack
            if probs.shape[1] == 2:
                return probs[:, 1]
            # Multi-class: 1 - P(benign)
            return 1.0 - probs[:, 0]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Section 4.3 — Energy-Aware Handshake Scheduler
# =============================================================================

class EnergyAwareScheduler:
    """
    Adaptive ML-KEM parameter selector.

    Section 4.3: couples device battery state (0–1) and threat score (0–1)
    to ML-KEM security parameter selection.

    Decision table (from Section 4.3, Fig. 3):
        battery < 0.20                          → ML-KEM-512  (conserve energy)
        battery 0.20–0.50 and threat < 0.70     → ML-KEM-768  (balanced)
        battery > 0.50 and threat ≥ 0.70        → ML-KEM-1024 (high security)
        battery > 0.50 and threat < 0.70        → ML-KEM-768  (balanced)
        battery < 0.50 and threat ≥ 0.70        → ML-KEM-768  (compromise)
    """

    THRESHOLDS = SCHEDULER["battery_thresholds"]

    def select_kem(self, battery: float, threat: float) -> str:
        """
        Select ML-KEM parameter set given battery level and threat score.

        Args:
            battery: residual battery fraction, [0.0, 1.0]
            threat:  threat score from ThreatClassifier, [0.0, 1.0]

        Returns:
            One of "ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"
        """
        if battery < self.THRESHOLDS["low"]:           # < 20%
            return "ML-KEM-512"
        elif battery < self.THRESHOLDS["medium"]:      # 20–50%
            return "ML-KEM-768"
        else:                                           # > 50%
            if threat >= SCHEDULER["threat_thresholds"]["medium"]:
                return "ML-KEM-1024"
            return "ML-KEM-768"

    def get_energy(self, kem_name: str) -> float:
        """Return per-handshake energy in mJ for the chosen parameter set."""
        return ML_KEM[kem_name]["energy_mJ"]

    def weighted_avg_energy(self, weights: dict = None) -> float:
        """
        Compute weighted average energy across parameter sets.
        Default weights are equal (1/3 each), producing 4.749 mJ.
        """
        if weights is None:
            weights = SCHEDULER["energy_weights"]
        return sum(
            weights[k] * ML_KEM[k]["energy_mJ"] for k in weights
        )


# =============================================================================
# Section 4.2 — Certificate Compression Model
# =============================================================================

class DeltaEncodingCompressor:
    """
    Models the delta-encoding certificate compression scheme.

    Section 4.2: exploits structural redundancy in hierarchical ML-DSA
    certificate chains (shared issuer fields, algorithm identifiers,
    predictable signature coefficient distributions).

    This class provides a Python-level simulation of the C implementation
    that runs on the STM32F429ZI MCU.
    """

    # Field sizes stripped during delta encoding (bytes)
    SHARED_FIELDS = {
        "issuer_dn": 128,
        "algorithm_id": 32,
        "validity_not_before_format": 15,
    }

    def compress_chain(self, certs: list) -> dict:
        """
        Simulate delta encoding + Zstandard compression of a certificate chain.

        Args:
            certs: list of dicts, each with 'param_set' and 'full_der_bytes'

        Returns:
            dict with original_bytes, delta_stripped_bytes, compressed_bytes,
            compression_ratio
        """
        orig = sum(c["full_der_bytes"] for c in certs)
        delta_stripped = orig
        for c in certs[1:]:  # root CA fields not stripped
            delta_stripped -= sum(self.SHARED_FIELDS.values())

        # Zstandard entropy coding (from paper: ML-DSA-65 → 1805 bytes after zstd)
        ratio = CERTIFICATE_COMPRESSION["compression_ratio"]
        compressed = int(delta_stripped * (1.0 - ratio))

        return {
            "original_bytes": orig,
            "delta_stripped_bytes": delta_stripped,
            "compressed_bytes": compressed,
            "compression_ratio": ratio,
        }

    def fits_in_ram(self, compressed_bytes: int, ram_limit: int = 262144) -> bool:
        """Check if compressed chain fits in 256 KB MCU RAM."""
        return compressed_bytes <= ram_limit


# =============================================================================
# Section 6.2 — Handshake Performance Simulator
# =============================================================================

class QPKIHandshakeSimulator:
    """
    Simulates the QPKI-SmartHome handshake on STM32F429ZI (ARM Cortex-M4).

    Section 6.2: models computation latency, energy consumption, and peak
    RAM given chosen ML-KEM and ML-DSA parameter sets.

    Reported paper values (Table 4):
        Computation latency : 38 ms
        End-to-end latency  : 545.6 ms  (including 250 kbps IEEE 802.15.4)
        Energy              : 4.749 mJ  (weighted average)
        Peak RAM            : 50 KB
        Auth success rate   : 99.7%
    """

    CLOCK_HZ = 168e6              # ARM Cortex-M4 @ 168 MHz
    WIRELESS_BPS = 250_000        # IEEE 802.15.4
    VOLTAGE_V = 3.3               # MCU supply voltage
    CURRENT_MA = 87.0             # active draw at 168 MHz

    def simulate(self, kem_name: str = "ML-KEM-768",
                 dsa_name: str = "ML-DSA-44") -> dict:
        """
        Compute handshake metrics for given parameter combination.

        Returns dict matching Table 4 of the paper.
        """
        kem = ML_KEM[kem_name]
        dsa = ML_DSA[dsa_name]

        # Computation latency approximated from reported 38 ms for ML-KEM-768 + ML-DSA-44
        # Scale factor derived from key sizes
        base_lat_ms = PERFORMANCE_RESULTS["computation_latency_ms"]
        scale_kem = kem["pk_bytes"] / ML_KEM["ML-KEM-768"]["pk_bytes"]
        scale_dsa = dsa["sig_bytes"] / ML_DSA["ML-DSA-44"]["sig_bytes"]
        comp_ms = base_lat_ms * (0.6 * scale_kem + 0.4 * scale_dsa)

        # Wireless transmission latency
        tx_bytes = (kem["ct_bytes"] + dsa["sig_bytes"]
                    + CERTIFICATE_COMPRESSION["after_zstd_bytes"])
        tx_ms = (tx_bytes * 8) / self.WIRELESS_BPS * 1000

        # End-to-end latency
        e2e_ms = comp_ms + tx_ms

        # Energy
        energy_mJ = kem["energy_mJ"]  # from measured per-set values in paper

        # Peak RAM
        peak_ram = kem["pk_bytes"] + kem["ct_bytes"] + dsa["sig_bytes"]

        return {
            "kem": kem_name,
            "dsa": dsa_name,
            "computation_latency_ms": round(comp_ms, 1),
            "transmission_latency_ms": round(tx_ms, 1),
            "end_to_end_latency_ms": round(e2e_ms, 1),
            "energy_mJ": energy_mJ,
            "peak_ram_bytes": peak_ram,
            "auth_success_rate": PERFORMANCE_RESULTS["auth_success_rate"],
        }


# =============================================================================
# Custom Loss — Focal Loss for imbalanced IoT traffic
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in CIC-IoT-2023.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return focal_loss.mean()


if __name__ == "__main__":
    # Quick smoke test
    model = ThreatClassifier(input_dim=20, n_classes=2)
    print(f"ThreatClassifier parameters: {model.parameter_count():,}")

    x = torch.randn(8, 20)
    logits = model(x)
    print(f"Logits shape: {logits.shape}")

    scheduler = EnergyAwareScheduler()
    for batt in [0.10, 0.35, 0.80]:
        for threat in [0.20, 0.85]:
            kem = scheduler.select_kem(batt, threat)
            e   = scheduler.get_energy(kem)
            print(f"  battery={batt:.0%} threat={threat:.0%} → {kem}  {e} mJ")

    sim = QPKIHandshakeSimulator()
    result = sim.simulate("ML-KEM-768", "ML-DSA-44")
    print(f"\nHandshake simulation: {result}")
