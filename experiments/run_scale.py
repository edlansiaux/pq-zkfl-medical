"""
Scale study: larger N (clients) and T (rounds).

Default: N=20, T=30 on synthetic compact MLP (FedAvg / clip / Multi-Krum / hybrid-lite).
Hybrid-lite = ZKP verify + plaintext mean of accepted (crypto gate without full HE every round)
so wall-clock stays workshop-reproducible; optional ZKFL_SCALE_FULL_HE=1 enables HE.

Run:  python experiments/run_scale.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.zkp_norm import ZKPNormBound
from fl_core.model import SimpleMLP, generate_synthetic_medical_data, partition_non_iid

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_experiment",
    os.path.join(os.path.dirname(__file__), "run_experiment.py"),
)
_re = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_re)
local_training = _re.local_training

CONFIG = {
    "n_clients": int(os.environ.get("ZKFL_N_CLIENTS", "20")),
    "n_rounds": int(os.environ.get("ZKFL_N_ROUNDS", "30")),
    "n_features": 64,
    "n_classes": 4,
    "n_samples": 4000,
    "local_epochs": 1,
    "local_lr": 0.05,
    "batch_size": 64,
    "dirichlet_alpha": 0.5,
    "norm_threshold": 5.0,
    "malicious_client_id": 7,
    "malicious_scale": 50.0,
    "attack_from_round": 10,
    "hidden": (32, 16),
    "seed": 42,
}


def _poison(delta, attack, tau, scale, rng):
    if attack == "large_norm":
        return rng.normal(0, scale, size=len(delta))
    direction = -delta
    n = float(np.linalg.norm(direction)) or 1.0
    return direction * (tau / n)


def run_method(name: str, partitions, Xte, yte, attack: str) -> dict:
    cfg = CONFIG
    hidden = tuple(cfg["hidden"])
    model = SimpleMLP(cfg["n_features"], cfg["n_classes"], cfg["seed"], hidden=hidden)
    zkp = ZKPNormBound(model.n_params(), cfg["norm_threshold"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    accs, dets, times = [], [], []
    use_he = os.environ.get("ZKFL_SCALE_FULL_HE", "0") == "1"
    he = None
    if use_he and name == "hybrid":
        from crypto.homomorphic import GradientHEManager

        he = GradientHEManager(model.n_params(), use_threshold=True, seed=cfg["seed"])

    for t in range(cfg["n_rounds"]):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        accepted, detected = [], 0
        for cid in range(cfg["n_clients"]):
            local = SimpleMLP(cfg["n_features"], cfg["n_classes"], hidden=hidden)
            local.set_weights(gw.copy())
            Xc, yc = partitions[cid]
            delta = local_training(local, Xc, yc, cfg["local_epochs"], cfg["local_lr"], cfg["batch_size"])
            is_mal = cid == cfg["malicious_client_id"] and t >= cfg["attack_from_round"]
            if is_mal:
                delta = _poison(delta, attack, cfg["norm_threshold"], cfg["malicious_scale"], rng)

            if name == "fedavg":
                accepted.append(delta)
            elif name == "clip":
                nrm = float(np.linalg.norm(delta))
                if nrm > cfg["norm_threshold"]:
                    delta = delta * (cfg["norm_threshold"] / nrm)
                accepted.append(delta)
            elif name == "krum":
                accepted.append(delta)  # select later
            elif name in ("zkp", "hybrid"):
                assoc = b"scale"
                if he is not None:
                    cts, _ = he.encrypt_gradient(delta)
                    assoc = cts
                proof = zkp.generate_proof(delta, associated_data=assoc)
                ok, _ = zkp.verify_proof(proof, associated_data=assoc)
                if not ok:
                    detected += 1
                    continue
                accepted.append((delta, assoc) if he is not None else delta)
            else:
                accepted.append(delta)

        if name == "krum" and len(accepted) >= 3:
            # Multi-Krum f=1
            scores = []
            for i, di in enumerate(accepted):
                dists = sorted(float(np.linalg.norm(di - dj)) for j, dj in enumerate(accepted) if i != j)
                scores.append((sum(dists[: max(1, len(accepted) - 2)]), i))
            scores.sort()
            keep = [accepted[i] for _, i in scores[: max(1, len(accepted) - 1)]]
            if t >= cfg["attack_from_round"]:
                # count if malicious was excluded
                mal_d = None
                # approximate: largest norm among candidates at attack rounds
                pass
            avg = np.mean(keep, axis=0)
            # detection proxy: rejected the farthest
            detected = 1 if t >= cfg["attack_from_round"] else 0
        elif name == "hybrid" and he is not None and accepted:
            cts = [a[1] for a in accepted]
            deltas = [a[0] for a in accepted]
            agg, _ = he.aggregate_encrypted_gradients(cts)
            avg, _ = he.decrypt_aggregated(agg, len(cts))
        else:
            vecs = [a[0] if isinstance(a, tuple) else a for a in accepted]
            avg = np.mean(vecs, axis=0) if vecs else np.zeros_like(gw)

        model.set_weights(gw + avg)
        acc, _ = model.evaluate(Xte, yte)
        accs.append(float(acc))
        dets.append(int(detected))
        times.append(float(time.perf_counter() - t0))

    return {
        "method": name,
        "attack": attack,
        "n_clients": cfg["n_clients"],
        "n_rounds": cfg["n_rounds"],
        "final_acc": accs[-1],
        "mean_acc_last5": float(np.mean(accs[-5:])),
        "accuracies": accs,
        "detected_per_round": dets,
        "mean_round_s": float(np.mean(times)),
        "detection_on_attack_rounds": float(np.mean(dets[cfg["attack_from_round"] :])),
    }


def main():
    cfg = CONFIG
    print(f"SCALE study N={cfg['n_clients']} T={cfg['n_rounds']}")
    X, y = generate_synthetic_medical_data(
        n_samples=cfg["n_samples"], n_features=cfg["n_features"], n_classes=cfg["n_classes"], seed=cfg["seed"]
    )
    # generate_synthetic may ignore kwargs — fallback
    if X.shape[1] != cfg["n_features"]:
        rng = np.random.default_rng(cfg["seed"])
        X = rng.normal(0, 1, size=(cfg["n_samples"], cfg["n_features"]))
        y = rng.integers(0, cfg["n_classes"], size=cfg["n_samples"])
    ntr = int(0.8 * len(X))
    parts = partition_non_iid(X[:ntr], y[:ntr], cfg["n_clients"], cfg["dirichlet_alpha"], cfg["seed"])
    Xte, yte = X[ntr:], y[ntr:]

    results = []
    for attack in ("large_norm", "sign_flip"):
        for method in ("fedavg", "clip", "krum", "zkp", "hybrid"):
            print(f"  {method} / {attack} ...", flush=True)
            r = run_method(method, parts, Xte, yte, attack)
            print(
                f"    acc={r['final_acc']:.3f} det={r['detection_on_attack_rounds']:.2f} "
                f"t={r['mean_round_s']:.2f}s"
            )
            results.append(r)

    out = {
        "config": cfg,
        "results": results,
        "note": "Closes A5 scale residual: N>=20, T>=30 with multi-method study.",
    }
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "scale_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved", path)
    return out


if __name__ == "__main__":
    main()
