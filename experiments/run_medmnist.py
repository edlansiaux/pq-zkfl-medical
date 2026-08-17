"""
MedMNIST medical imaging demo (PneumoniaMNIST) under the target protocol stack.

Falls back to UCI Breast Cancer if medmnist is not installed / offline.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.homomorphic import HE_CLAIMED_SECURITY_BITS, HE_N, HE_PRESET, GradientHEManager
from crypto.qrom_nizk import UnruhNormNIZK
from fl_core.model import SimpleMLP, load_medical_dataset, partition_non_iid
from experiments.run_experiment import local_training


def main():
    dataset = os.environ.get("ZKFL_DATASET", "pneumoniamnist")
    try:
        X, y, meta = load_medical_dataset(dataset, seed=42)
    except Exception as e:  # noqa: BLE001
        print(f"MedMNIST unavailable ({e}); falling back to breast_cancer")
        X, y, meta = load_medical_dataset("breast_cancer", seed=42)

    print("Dataset:", meta)
    print(f"HE preset={HE_PRESET} n={HE_N} claimed_bits={HE_CLAIMED_SECURITY_BITS}")

    # Subsample / compact model for HE tractability
    n_features = meta["n_features"]
    n_classes = meta["n_classes"]
    if n_features > 200:
        # PCA-free downsample: random projection
        rng = np.random.default_rng(0)
        P = rng.normal(0, 1.0 / np.sqrt(64), size=(n_features, 64))
        X = X @ P
        n_features = 64
        meta = {**meta, "n_features": 64, "projected": True}

    n_train = int(0.8 * len(X))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    parts = partition_non_iid(Xtr, ytr, 5, 0.5, 42)

    hidden = (32, 16)
    model = SimpleMLP(n_features, n_classes, 42, hidden=hidden)
    n_params = model.n_params()
    print(f"n_params={n_params} chunks={(n_params + HE_N - 1) // HE_N}")

    he = GradientHEManager(n_params, use_threshold=True, threshold=2, threshold_parties=3, seed=42)
    # r=32 for imaging demo latency; production Unruh default is 128
    zkp = UnruhNormNIZK(n_params, 8.0, reps=32, seed=42)

    metrics = {"accuracies": [], "detected": [], "dataset": meta, "he_preset": HE_PRESET}
    rng = np.random.default_rng(42)

    for round_t in range(3):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        cts_all, det = [], 0
        for cid in range(5):
            local = SimpleMLP(n_features, n_classes, hidden=hidden)
            local.set_weights(gw.copy())
            delta = local_training(local, parts[cid][0], parts[cid][1], 1, 0.05, 32)
            if cid == 3 and round_t >= 2:
                delta = rng.normal(0, 500.0, size=len(delta))
            he_cts, _ = he.encrypt_gradient(delta)
            proof = zkp.generate_proof(delta, associated_data=he_cts)
            ok, _ = zkp.verify_proof(proof, associated_data=he_cts)
            if not ok:
                det += 1
                continue
            cts_all.append(he_cts)
        if cts_all:
            agg, _ = he.aggregate_encrypted_gradients(cts_all)
            mean, _ = he.decrypt_aggregated(agg, len(cts_all))
            model.set_weights(gw + mean)
        acc, _ = model.evaluate(Xte, yte)
        metrics["accuracies"].append(float(acc))
        metrics["detected"].append(int(det))
        print(f"round {round_t+1}: acc={acc:.4f} detected={det} time={time.perf_counter()-t0:.1f}s")

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "medmnist_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", out)
    return metrics


if __name__ == "__main__":
    main()
