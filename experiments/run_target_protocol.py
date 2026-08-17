"""
Target-protocol demo closing the former roadmap gaps:
  - Real medical data (UCI Breast Cancer)
  - Full-vector HE (all parameter chunks) with Classic-128-oriented n
  - (t,n)-threshold BFV decryption (no single sk holder)
  - Unruh-style QROM-oriented NIZK bound to ciphertext

Run:  python experiments/run_target_protocol.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.homomorphic import HE_CLAIMED_SECURITY_BITS, HE_N, GradientHEManager
from crypto.ml_kem import MLKEM768
from crypto.qrom_nizk import UnruhNormNIZK
from fl_core.model import SimpleMLP, load_medical_dataset, partition_non_iid
from experiments.run_experiment import local_training

CONFIG = {
    "dataset": "breast_cancer",
    "n_clients": 5,
    "n_rounds": 3,
    "local_epochs": 2,
    "local_lr": 0.05,
    "batch_size": 32,
    "dirichlet_alpha": 0.5,
    "norm_threshold": 8.0,
    "malicious_client_id": 3,
    "malicious_scale": 500.0,
    "seed": 42,
    "hidden": (32, 16),
    "unruh_reps": 16,  # raise for higher soundness
    "threshold_parties": 3,
    "threshold": 2,
}


def main():
    print("=" * 60)
    print("ZKFL-PQ TARGET PROTOCOL DEMO")
    print(f"HE_N={HE_N}, security_target≈{HE_CLAIMED_SECURITY_BITS}-bit class")
    print("=" * 60)

    X, y, meta = load_medical_dataset(CONFIG["dataset"], CONFIG["seed"])
    print(f"Dataset: {meta}")

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    parts = partition_non_iid(
        X_train,
        y_train,
        CONFIG["n_clients"],
        CONFIG["dirichlet_alpha"],
        CONFIG["seed"],
    )

    n_features = meta["n_features"]
    n_classes = meta["n_classes"]
    hidden = tuple(CONFIG["hidden"])

    model = SimpleMLP(n_features, n_classes, CONFIG["seed"], hidden=hidden)
    n_params = model.n_params()
    print(f"Model params (FULL vector encrypted): {n_params}, chunks={ (n_params + HE_N - 1)//HE_N }")

    he = GradientHEManager(
        n_params,
        scale=100.0,
        seed=CONFIG["seed"],
        threshold_parties=CONFIG["threshold_parties"],
        threshold=CONFIG["threshold"],
        use_threshold=True,
    )
    assert he.sk is None and he.threshold_engine is not None
    print(
        f"Threshold BFV: ({CONFIG['threshold']},{CONFIG['threshold_parties']}) "
        f"— server holds NO monolithic sk"
    )

    zkp = UnruhNormNIZK(
        n_params, CONFIG["norm_threshold"], reps=CONFIG["unruh_reps"], seed=CONFIG["seed"]
    )
    print(f"Unruh NIZK reps={CONFIG['unruh_reps']} (QROM-oriented)")

    kem = MLKEM768(seed=CONFIG["seed"])
    ek, _, _ = kem.keygen()

    metrics = {
        "accuracies": [],
        "round_times": [],
        "detected": [],
        "msg_kb": [],
        "dataset": meta,
        "he_n": HE_N,
        "he_security_target_bits": HE_CLAIMED_SECURITY_BITS,
        "threshold": CONFIG["threshold"],
        "threshold_parties": CONFIG["threshold_parties"],
        "unruh_reps": CONFIG["unruh_reps"],
        "n_params": n_params,
        "full_vector_he": True,
        "single_decryptor": False,
    }

    rng = np.random.default_rng(CONFIG["seed"])

    for round_t in range(CONFIG["n_rounds"]):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        all_cts, accepted = [], []
        detected = 0
        msg = 0

        for cid in range(CONFIG["n_clients"]):
            local = SimpleMLP(n_features, n_classes, hidden=hidden)
            local.set_weights(gw.copy())
            Xc, yc = parts[cid]
            delta = local_training(
                local,
                Xc,
                yc,
                CONFIG["local_epochs"],
                CONFIG["local_lr"],
                CONFIG["batch_size"],
            )
            is_mal = cid == CONFIG["malicious_client_id"] and round_t >= 2
            if is_mal:
                delta = rng.normal(0, CONFIG["malicious_scale"], size=len(delta))

            # Full-vector HE
            he_cts, _ = he.encrypt_gradient(delta)
            proof = zkp.generate_proof(delta, associated_data=he_cts)
            ok, _ = zkp.verify_proof(proof, associated_data=he_cts)
            if not ok:
                detected += 1
                print(f"  round {round_t+1}: reject client {cid} (mal={is_mal})")
                continue
            ct_kem, _, _ = kem.encaps(ek)
            all_cts.append(he_cts)
            accepted.append(delta)
            msg += (
                proof["proof_size_bytes"]
                + ct_kem["u"].nbytes
                + ct_kem["v"].nbytes
                + sum(
                    np.asarray(c["c0"]).nbytes + np.asarray(c["c1"]).nbytes
                    if np.asarray(c["c0"]).dtype != object
                    else 16 * (len(c["c0"]) + len(c["c1"]))
                    for c in he_cts
                )
            )

        if all_cts:
            agg, _ = he.aggregate_encrypted_gradients(all_cts)
            # Threshold decrypt — reconstructs only with t shares
            he_mean, _ = he.decrypt_aggregated(agg, len(all_cts))
            model.set_weights(gw + he_mean)

        acc, loss = model.evaluate(X_test, y_test)
        rt = time.perf_counter() - t0
        metrics["accuracies"].append(float(acc))
        metrics["round_times"].append(float(rt))
        metrics["detected"].append(int(detected))
        metrics["msg_kb"].append(msg / 1024)
        print(
            f"Round {round_t+1}: acc={acc:.4f} loss={loss:.4f} "
            f"time={rt:.2f}s detected={detected} msg={msg/1024:.1f}KB"
        )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "target_protocol_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {path}")
    print(
        f"FINAL acc={metrics['accuracies'][-1]:.4f} "
        f"mean_time={np.mean(metrics['round_times']):.2f}s "
        f"threshold_decrypt=ON unruh=ON full_vector_HE=ON medical={meta['name']}"
    )
    return metrics


if __name__ == "__main__":
    main()
