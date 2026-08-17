"""
Multi-baseline + multi-seed evaluation (synthetic large-norm and sign-flip).

Baselines: FedAvg, Clip@tau, Multi-Krum, HE-only, ZKP-only, Hybrid.
Attacks: large_norm, sign_flip (norm == tau).
Seeds: 42..46
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.homomorphic import HE_N, GradientHEManager
from crypto.ml_kem import MLKEM768
from crypto.zkp_norm import ZKPNormBound
from fl_core.model import (
    SimpleMLP,
    generate_synthetic_medical_data,
    partition_non_iid,
)

# Avoid package import of run_experiment when launched as a script
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_experiment",
    os.path.join(os.path.dirname(__file__), "run_experiment.py"),
)
_re = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_re)
CONFIG = _re.CONFIG
local_training = _re.local_training

SEEDS = [42, 43, 44, 45, 46]
ATTACKS = ["large_norm", "sign_flip"]


def _poison(delta: np.ndarray, attack: str, tau: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    if attack == "large_norm":
        return rng.normal(0, scale, size=len(delta))
    # sign-flip / directional: opposite of honest update, scaled to exactly tau
    direction = -delta
    n = float(np.linalg.norm(direction))
    if n < 1e-12:
        direction = rng.normal(0, 1, size=len(delta))
        n = float(np.linalg.norm(direction))
    return direction * (tau / n)


def _collect_deltas(
    model: SimpleMLP,
    partitions,
    config: dict,
    attack: str,
    round_t: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[bool]]:
    global_weights = model.get_weights().copy()
    deltas = []
    malicious_flags = []
    for client_id in range(config["n_clients"]):
        local_model = SimpleMLP(config["n_features"], config["n_classes"])
        local_model.set_weights(global_weights.copy())
        X_c, y_c = partitions[client_id]
        delta = local_training(
            local_model,
            X_c,
            y_c,
            config["local_epochs"],
            config["local_lr"],
            config["batch_size"],
        )
        is_mal = client_id == config["malicious_client_id"] and round_t >= 3
        if is_mal:
            delta = _poison(delta, attack, config["norm_threshold"], config["malicious_scale"], rng)
        deltas.append(delta)
        malicious_flags.append(is_mal)
    return deltas, malicious_flags


def run_fedavg(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    rng = np.random.default_rng(config["seed"])
    times, accs = [], []
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, _ = _collect_deltas(model, partitions, config, attack, round_t, rng)
        avg = np.mean(deltas, axis=0)
        model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    return {"final_accuracy": accs[-1], "mean_round_time": float(np.mean(times)), "accuracies": accs, "detection_rate": 0.0}


def run_clip(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    rng = np.random.default_rng(config["seed"])
    tau = config["norm_threshold"]
    times, accs = [], []
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, _ = _collect_deltas(model, partitions, config, attack, round_t, rng)
        clipped = []
        for d in deltas:
            n = float(np.linalg.norm(d))
            clipped.append(d if n <= tau else d * (tau / n))
        avg = np.mean(clipped, axis=0)
        model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    return {"final_accuracy": accs[-1], "mean_round_time": float(np.mean(times)), "accuracies": accs, "detection_rate": 0.0}


def multi_krum_aggregate(deltas: List[np.ndarray], f: int = 1) -> np.ndarray:
    """Select n-f-2 candidates with smallest Multi-Krum scores, then mean."""
    n = len(deltas)
    if n <= 2 * f + 2:
        return np.mean(deltas, axis=0)
    scores = []
    for i in range(n):
        dists = sorted(float(np.linalg.norm(deltas[i] - deltas[j])) for j in range(n) if j != i)
        # sum of distances to n-f-2 nearest neighbours
        k = max(1, n - f - 2)
        scores.append(sum(dists[:k]))
    order = np.argsort(scores)
    keep = order[: max(1, n - f)]
    return np.mean([deltas[i] for i in keep], axis=0)


def run_krum(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    rng = np.random.default_rng(config["seed"])
    times, accs = [], []
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, _ = _collect_deltas(model, partitions, config, attack, round_t, rng)
        avg = multi_krum_aggregate(deltas, f=1)
        model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    return {"final_accuracy": accs[-1], "mean_round_time": float(np.mean(times)), "accuracies": accs, "detection_rate": 0.0}


def run_he_only(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    n_params = model.n_params()
    prot = min(HE_N, n_params)
    he = GradientHEManager(prot, scale=100.0, seed=config["seed"])
    rng = np.random.default_rng(config["seed"])
    times, accs = [], []
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, _ = _collect_deltas(model, partitions, config, attack, round_t, rng)
        cts = []
        for d in deltas:
            ct, _ = he.encrypt_gradient(d[:prot])
            cts.append(ct)
        agg, _ = he.aggregate_encrypted_gradients(cts)
        he_res, _ = he.decrypt_aggregated(agg, len(cts))
        avg = np.zeros(n_params)
        avg[:prot] = he_res
        model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    return {"final_accuracy": accs[-1], "mean_round_time": float(np.mean(times)), "accuracies": accs, "detection_rate": 0.0}


def run_zkp_only(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    n_params = model.n_params()
    prot = min(HE_N, n_params)
    zkp = ZKPNormBound(prot, config["norm_threshold"], seed=config["seed"])
    rng = np.random.default_rng(config["seed"])
    times, accs = [], []
    detected_mal, total_mal = 0, 0
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, flags = _collect_deltas(model, partitions, config, attack, round_t, rng)
        accepted = []
        for d, is_mal in zip(deltas, flags):
            if is_mal:
                total_mal += 1
            protected = d[:prot].copy()
            # Bind to a deterministic placeholder (no HE) so FS still hashes bytes
            dummy_ct = protected.tobytes()[:64]
            proof = zkp.generate_proof(protected, associated_data=dummy_ct)
            ok, _ = zkp.verify_proof(proof, associated_data=dummy_ct)
            if ok:
                accepted.append(d)
            elif is_mal:
                detected_mal += 1
        if accepted:
            avg = np.mean(accepted, axis=0)
            model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    det = detected_mal / total_mal if total_mal else 0.0
    return {
        "final_accuracy": accs[-1],
        "mean_round_time": float(np.mean(times)),
        "accuracies": accs,
        "detection_rate": det,
    }


def run_hybrid_median(partitions, test_X, test_y, config, attack: str) -> dict:
    """Default composition ablation: ZKP gate + coordinate-wise median (no HE)."""
    from fl_core.robust_agg import coord_median

    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    n_params = model.n_params()
    prot = min(HE_N, n_params)
    zkp = ZKPNormBound(prot, config["norm_threshold"], seed=config["seed"])
    rng = np.random.default_rng(config["seed"])
    times, accs = [], []
    detected_mal, total_mal = 0, 0
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, flags = _collect_deltas(model, partitions, config, attack, round_t, rng)
        accepted = []
        for d, is_mal in zip(deltas, flags):
            if is_mal:
                total_mal += 1
            protected = d[:prot].copy()
            dummy_ct = protected.tobytes()[:64]
            proof = zkp.generate_proof(protected, associated_data=dummy_ct)
            ok, _ = zkp.verify_proof(proof, associated_data=dummy_ct)
            if ok:
                accepted.append(d)
            elif is_mal:
                detected_mal += 1
        if accepted:
            model.set_weights(model.get_weights() + coord_median(accepted))
        times.append(time.perf_counter() - t0)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    det = detected_mal / total_mal if total_mal else 0.0
    return {
        "final_accuracy": accs[-1],
        "mean_round_time": float(np.mean(times)),
        "accuracies": accs,
        "detection_rate": det,
    }


def run_hybrid(partitions, test_X, test_y, config, attack: str) -> dict:
    model = SimpleMLP(config["n_features"], config["n_classes"], config["seed"])
    n_params = model.n_params()
    prot = min(HE_N, n_params)
    he = GradientHEManager(prot, scale=100.0, seed=config["seed"])
    zkp = ZKPNormBound(prot, config["norm_threshold"], seed=config["seed"])
    kem = MLKEM768(seed=config["seed"])
    ek, _, _ = kem.keygen()
    rng = np.random.default_rng(config["seed"])
    times, accs, msg_kb = [], [], []
    detected_mal, total_mal = 0, 0
    for round_t in range(config["n_rounds"]):
        t0 = time.perf_counter()
        deltas, flags = _collect_deltas(model, partitions, config, attack, round_t, rng)
        all_cts = []
        accepted_deltas = []
        total_msg = 0
        for d, is_mal in zip(deltas, flags):
            if is_mal:
                total_mal += 1
            protected = d[:prot].copy()
            he_cts, _ = he.encrypt_gradient(protected)
            proof = zkp.generate_proof(protected, associated_data=he_cts)
            ok, _ = zkp.verify_proof(proof, associated_data=he_cts)
            if not ok:
                if is_mal:
                    detected_mal += 1
                continue
            ct_kem, _, _ = kem.encaps(ek)
            all_cts.append(he_cts)
            accepted_deltas.append(d)
            total_msg += (
                proof["proof_size_bytes"]
                + ct_kem["u"].nbytes
                + ct_kem["v"].nbytes
                + sum(c["c0"].nbytes + c["c1"].nbytes for c in he_cts)
            )
        if all_cts:
            agg, _ = he.aggregate_encrypted_gradients(all_cts)
            he_res, _ = he.decrypt_aggregated(agg, len(all_cts))
            # Mean of accepted full deltas, overwrite protected with HE
            avg = np.mean(accepted_deltas, axis=0)
            avg[:prot] = he_res
            model.set_weights(model.get_weights() + avg)
        times.append(time.perf_counter() - t0)
        msg_kb.append(total_msg / 1024)
        acc, _ = model.evaluate(test_X, test_y)
        accs.append(acc)
    det = detected_mal / total_mal if total_mal else 0.0
    return {
        "final_accuracy": accs[-1],
        "mean_round_time": float(np.mean(times)),
        "mean_msg_kb": float(np.mean(msg_kb)) if msg_kb else 0.0,
        "accuracies": accs,
        "detection_rate": det,
    }


METHODS = {
    "fedavg": run_fedavg,
    "clip": run_clip,
    "multi_krum": run_krum,
    "he_only": run_he_only,
    "zkp_only": run_zkp_only,
    "hybrid": run_hybrid,
    "hybrid_median": run_hybrid_median,
}


def _summarize(runs: List[dict]) -> dict:
    acc = [r["final_accuracy"] for r in runs]
    det = [r["detection_rate"] for r in runs]
    t = [r["mean_round_time"] for r in runs]
    out = {
        "acc_mean": float(np.mean(acc)),
        "acc_std": float(np.std(acc)),
        "det_mean": float(np.mean(det)),
        "det_std": float(np.std(det)),
        "time_mean": float(np.mean(t)),
        "time_std": float(np.std(t)),
        "per_seed": runs,
    }
    if any("mean_msg_kb" in r for r in runs):
        m = [r.get("mean_msg_kb", 0.0) for r in runs]
        out["msg_kb_mean"] = float(np.mean(m))
        out["msg_kb_std"] = float(np.std(m))
    return out


def main():
    print("=" * 60)
    print("ZKFL-PQ baselines + attacks (5 seeds)")
    print("=" * 60)

    results: Dict = {}
    for attack in ATTACKS:
        results[attack] = {}
        for method, fn in METHODS.items():
            print(f"\n>>> {method} / {attack}")
            seed_runs = []
            for seed in SEEDS:
                cfg = CONFIG.copy()
                cfg["seed"] = seed
                cfg["n_rounds"] = 10
                X, y = generate_synthetic_medical_data(
                    cfg["n_samples"], cfg["n_features"], cfg["n_classes"], seed
                )
                n_train = int(0.8 * len(X))
                X_train, X_test = X[:n_train], X[n_train:]
                y_train, y_test = y[:n_train], y[n_train:]
                parts = partition_non_iid(
                    X_train, y_train, cfg["n_clients"], cfg["dirichlet_alpha"], seed
                )
                r = fn(parts, X_test, y_test, cfg, attack)
                r["seed"] = seed
                seed_runs.append(r)
                print(
                    f"  seed={seed}: acc={r['final_accuracy']:.4f} "
                    f"det={r['detection_rate']:.1%} time={r['mean_round_time']:.2f}s"
                )
            results[attack][method] = _summarize(seed_runs)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "baseline_results.json")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2)

    print("\n=== SUMMARY ===")
    for attack in ATTACKS:
        print(f"\n[{attack}]")
        for method in METHODS:
            s = results[attack][method]
            print(
                f"  {method:12s} acc={100*s['acc_mean']:.1f}±{100*s['acc_std']:.1f}%  "
                f"det={100*s['det_mean']:.0f}%  t={s['time_mean']:.2f}s"
            )
    print(f"\nSaved {path}")
    return results


if __name__ == "__main__":
    main()
