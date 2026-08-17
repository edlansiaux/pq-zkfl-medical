"""
Full-resolution MedMNIST (PneumoniaMNIST) — no feature projection.

Uses native 28x28 = 784 pixels, compact MLP, Unruh + full-vector HE + threshold.
Subsample clients/rounds for CPU; images are NOT projected to 64-D.

Run:  python experiments/run_medmnist_fullres.py
Env:  ZKFL_DATASET=pneumoniamnist (default)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.enc_consistency import EncConsistencyGadget, bind_associated_data
from crypto.homomorphic import HE_DELTA, HE_N, HE_Q, GradientHEManager
from crypto.qrom_nizk import UnruhNormNIZK
from fl_core.model import SimpleMLP, load_medical_dataset, partition_non_iid

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_experiment",
    os.path.join(os.path.dirname(__file__), "run_experiment.py"),
)
_re = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_re)
local_training = _re.local_training

CONFIG = {
    "dataset": os.environ.get("ZKFL_DATASET", "pneumoniamnist"),
    "n_clients": 5,
    "n_rounds": 3,
    "local_epochs": 1,
    "local_lr": 0.02,
    "batch_size": 32,
    "dirichlet_alpha": 0.5,
    "norm_threshold": 12.0,
    "malicious_client_id": 2,
    "malicious_scale": 500.0,
    "hidden": (16, 8),
    "unruh_reps": int(os.environ.get("ZKFL_UNRUH_REPS", "8")),
    "seed": 42,
    "max_samples": 800,
}


def main():
    cfg = CONFIG
    print("=" * 60)
    print("MedMNIST FULL-RESOLUTION (no projection)")
    print("=" * 60)
    try:
        X, y, meta = load_medical_dataset(cfg["dataset"], cfg["seed"])
    except Exception as e:  # noqa: BLE001
        print(f"MedMNIST unavailable ({e}); using breast_cancer fallback is NOT full-res vision.")
        raise

    # Cap samples but keep full feature dim
    rng = np.random.default_rng(cfg["seed"])
    if len(X) > cfg["max_samples"]:
        idx = rng.choice(len(X), size=cfg["max_samples"], replace=False)
        X, y = X[idx], y[idx]
        meta = {**meta, "n_samples": len(X), "full_resolution": True, "projected": False}

    assert meta.get("projected") is not True
    assert int(meta["n_features"]) >= 700, f"expected full-res imaging features, got {meta}"
    print("Dataset:", meta)

    ntr = int(0.8 * len(X))
    parts = partition_non_iid(X[:ntr], y[:ntr], cfg["n_clients"], cfg["dirichlet_alpha"], cfg["seed"])
    Xte, yte = X[ntr:], y[ntr:]

    hidden = tuple(cfg["hidden"])
    model = SimpleMLP(meta["n_features"], meta["n_classes"], cfg["seed"], hidden=hidden)
    n_params = model.n_params()
    chunks = (n_params + HE_N - 1) // HE_N
    print(f"n_params={n_params} full-res features={meta['n_features']} HE_chunks={chunks}")

    he = GradientHEManager(n_params, use_threshold=True, threshold=2, threshold_parties=3, seed=cfg["seed"])
    enc = EncConsistencyGadget(seed=cfg["seed"] + 3)
    zkp = UnruhNormNIZK(n_params, cfg["norm_threshold"], reps=cfg["unruh_reps"], seed=cfg["seed"])

    metrics = {
        "dataset": meta,
        "n_params": n_params,
        "he_chunks": chunks,
        "projected": False,
        "full_resolution": True,
        "unruh_reps": cfg["unruh_reps"],
        "accuracies": [],
        "detected": [],
        "round_times": [],
    }

    for t in range(cfg["n_rounds"]):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        cts_all, det = [], 0
        for cid in range(cfg["n_clients"]):
            local = SimpleMLP(meta["n_features"], meta["n_classes"], hidden=hidden)
            local.set_weights(gw.copy())
            delta = local_training(
                local, parts[cid][0], parts[cid][1], cfg["local_epochs"], cfg["local_lr"], cfg["batch_size"]
            )
            if cid == cfg["malicious_client_id"] and t >= cfg["n_rounds"] - 1:
                delta = rng.normal(0, cfg["malicious_scale"], size=len(delta))
            he_cts, coins, pts, _ = he.encrypt_gradient_with_coins(delta)
            enc_proof = enc.prove_gradient(he.pk, he_cts, pts, coins, he.bfv.n, HE_Q, HE_DELTA)
            enc_ok, _ = enc.verify_gradient(he.pk, he_cts, enc_proof)
            assoc = bind_associated_data(he_cts, enc_proof)
            proof = zkp.generate_proof(delta, associated_data=assoc)
            ok, _ = zkp.verify_proof(proof, associated_data=assoc)
            if not (ok and enc_ok):
                det += 1
                continue
            cts_all.append(he_cts)
        if cts_all:
            agg, _ = he.aggregate_encrypted_gradients(cts_all)
            mean, _ = he.decrypt_aggregated(agg, len(cts_all))
            model.set_weights(gw + mean)
        acc, _ = model.evaluate(Xte, yte)
        rt = time.perf_counter() - t0
        metrics["accuracies"].append(float(acc))
        metrics["detected"].append(int(det))
        metrics["round_times"].append(float(rt))
        print(f"round {t+1}: acc={acc:.4f} detected={det} time={rt:.1f}s chunks={chunks}")

    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results", "medmnist_fullres_results.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", path)
    return metrics


if __name__ == "__main__":
    main()
