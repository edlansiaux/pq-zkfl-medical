"""
PneumoniaMNIST with ConvNet28 — no projection, no compact MLP head.

Full imaging CNN + Unruh ZKP + full-vector HE + threshold + median.

Run:  python experiments/run_medmnist_cnn.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.enc_consistency import EncConsistencyGadget, bind_associated_data
from crypto.fused_he import create_he_manager
from crypto.homomorphic import HE_N, HE_Q, HE_DELTA
from crypto.qrom_nizk import UnruhNormNIZK
from fl_core.cnn import ConvNet28
from fl_core.model import load_medical_dataset, partition_non_iid
from fl_core.robust_agg import robust_aggregate

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
    "n_clients": int(os.environ.get("ZKFL_CNN_CLIENTS", "3")),
    "n_rounds": int(os.environ.get("ZKFL_CNN_ROUNDS", "2")),
    "local_epochs": 1,
    "local_lr": 0.01,
    "batch_size": 32,
    "dirichlet_alpha": 0.5,
    "norm_threshold": 25.0,
    "malicious_client_id": 1,
    "malicious_scale": 500.0,
    "unruh_reps": int(os.environ.get("ZKFL_UNRUH_REPS", "4")),
    "seed": 42,
    "max_samples": int(os.environ.get("ZKFL_CNN_SAMPLES", "600")),
    "fc": int(os.environ.get("ZKFL_CNN_FC", "64")),
}


def main():
    cfg = CONFIG
    print("=" * 60)
    print("MedMNIST ConvNet28 (no compact head, no projection)")
    print("=" * 60)
    X, y, meta = load_medical_dataset(cfg["dataset"], cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    if len(X) > cfg["max_samples"]:
        idx = rng.choice(len(X), size=cfg["max_samples"], replace=False)
        X, y = X[idx], y[idx]
    meta = {
        **meta,
        "n_samples": len(X),
        "full_resolution": True,
        "projected": False,
        "model": "ConvNet28",
        "compact_head": False,
    }
    assert int(meta["n_features"]) >= 700
    print("Dataset:", meta)

    ntr = int(0.8 * len(X))
    parts = partition_non_iid(
        X[:ntr], y[:ntr], cfg["n_clients"], cfg["dirichlet_alpha"], cfg["seed"]
    )
    Xte, yte = X[ntr:], y[ntr:]

    model = ConvNet28(meta["n_classes"], cfg["seed"], fc=cfg["fc"])
    n_params = model.n_params()
    chunks = (n_params + HE_N - 1) // HE_N
    print(f"n_params={n_params} HE_chunks={chunks} compact_head=False")

    he = create_he_manager(
        n_params, use_threshold=True, threshold=2, threshold_parties=3, seed=cfg["seed"]
    )
    enc = EncConsistencyGadget(seed=cfg["seed"] + 3)
    zkp = UnruhNormNIZK(n_params, cfg["norm_threshold"], reps=cfg["unruh_reps"], seed=cfg["seed"])

    metrics = {
        "dataset": meta,
        "n_params": n_params,
        "he_chunks": chunks,
        "projected": False,
        "compact_head": False,
        "model": "ConvNet28",
        "unruh_reps": cfg["unruh_reps"],
        "he_backend": getattr(he, "backend", type(he).__name__),
        "accuracies": [],
        "detected": [],
        "round_times": [],
        "msg_kb": [],
    }

    for rnd in range(cfg["n_rounds"]):
        t0 = time.perf_counter()
        accepted = []
        detected = 0
        msg = 0.0
        for cid, (Xi, yi) in enumerate(parts):
            delta = local_training(
                model, Xi, yi, cfg["local_epochs"], cfg["local_lr"], cfg["batch_size"]
            )
            is_mal = cid == cfg["malicious_client_id"] and rnd == cfg["n_rounds"] - 1
            if is_mal:
                delta = np.ones_like(delta) * cfg["malicious_scale"]

            if hasattr(he, "encrypt_gradient_with_coins"):
                he_cts, coins, pts, _ = he.encrypt_gradient_with_coins(delta)
                enc_proof = enc.prove_gradient(
                    he.pk, he_cts, pts, coins, he.bfv.n, HE_Q, HE_DELTA
                )
                enc_ok, _ = enc.verify_gradient(he.pk, he_cts, enc_proof)
                ad = bind_associated_data(he_cts, enc_proof)
                proof = zkp.generate_proof(delta, associated_data=ad)
                ok, _ = zkp.verify_proof(proof, associated_data=ad)
                ok = bool(ok and enc_ok)
            else:
                he_cts, _ = he.encrypt_gradient(delta)
                proof = zkp.generate_proof(delta, associated_data=he_cts)
                ok, _ = zkp.verify_proof(proof, associated_data=he_cts)
            if not ok:
                if is_mal:
                    detected += 1
                continue
            vec, _ = he.decrypt_aggregated(he_cts, 1)
            accepted.append(np.asarray(vec, dtype=np.float64).ravel()[:n_params])
            msg += float(proof.get("proof_size_bytes", 0)) / 1024.0
            for c in he_cts:
                msg += (c["c0"].nbytes + c["c1"].nbytes) / 1024.0

        if accepted:
            bar = robust_aggregate(accepted, method="median")
            model.set_weights(model.get_weights() + bar)
        acc, loss = model.evaluate(Xte, yte)
        dt = time.perf_counter() - t0
        metrics["accuracies"].append(float(acc))
        metrics["detected"].append(int(detected))
        metrics["round_times"].append(float(dt))
        metrics["msg_kb"].append(float(msg))
        print(
            f"Round {rnd+1}: acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s "
            f"detected={detected} msg={msg:.0f}KB params={n_params}"
        )

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "medmnist_cnn_results.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved", out)
    print(
        f"FINAL acc={metrics['accuracies'][-1]:.4f} n_params={n_params} "
        f"compact_head=False model=ConvNet28"
    )


if __name__ == "__main__":
    main()
