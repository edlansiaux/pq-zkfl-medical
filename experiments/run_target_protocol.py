"""
Target-protocol demo (default composition):
  - UCI medical data, full-vector HE, Enc-consistency, Unruh NIZK
  - Fused SEAL+threshold HE when TenSEAL is available (ZKFL_HE_BACKEND=fused)
  - Post-ZKP robust aggregation (default: coordinate-wise median)

Run:  python experiments/run_target_protocol.py
Env:  ZKFL_HE_BACKEND=fused|numpy|tenseal
      ZKFL_ROBUST_AGG=median|krum|mean
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.enc_consistency import EncConsistencyGadget, bind_associated_data
from crypto.fused_he import FusedSealThresholdHE, create_he_manager
from crypto.homomorphic import HE_CLAIMED_SECURITY_BITS, HE_DELTA, HE_N, HE_Q
from crypto.ml_kem import MLKEM768
from crypto.qrom_nizk import UnruhNormNIZK
from crypto.seal_backend import seal_available
from fl_core.model import SimpleMLP, load_medical_dataset, partition_non_iid
from fl_core.robust_agg import robust_aggregate
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
    "unruh_reps": 64,
    "threshold_parties": 3,
    "threshold": 2,
    "robust_f": 1,
}


def main():
    if not os.environ.get("ZKFL_HE_BACKEND"):
        os.environ["ZKFL_HE_BACKEND"] = "fused" if seal_available() else "numpy"
    if not os.environ.get("ZKFL_ROBUST_AGG"):
        os.environ["ZKFL_ROBUST_AGG"] = "median"

    backend = os.environ["ZKFL_HE_BACKEND"].lower()
    robust = os.environ["ZKFL_ROBUST_AGG"].lower()
    print("=" * 60)
    print("ZKFL-PQ TARGET PROTOCOL DEMO")
    print(f"HE_N={HE_N}, security_target~={HE_CLAIMED_SECURITY_BITS}-bit class")
    print(f"HE backend={backend} (tenseal available={seal_available()})")
    print(f"Robust agg={robust} (post-ZKP)")
    print("=" * 60)

    X, y, meta = load_medical_dataset(CONFIG["dataset"], CONFIG["seed"])
    print(f"Dataset: {meta}")

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    parts = partition_non_iid(
        X_train, y_train, CONFIG["n_clients"], CONFIG["dirichlet_alpha"], CONFIG["seed"]
    )

    n_features = meta["n_features"]
    n_classes = meta["n_classes"]
    hidden = tuple(CONFIG["hidden"])

    model = SimpleMLP(n_features, n_classes, CONFIG["seed"], hidden=hidden)
    n_params = model.n_params()
    print(
        f"Model params (FULL vector encrypted): {n_params}, "
        f"chunks={(n_params + HE_N - 1) // HE_N}"
    )

    he = create_he_manager(
        n_params,
        scale=100.0,
        seed=CONFIG["seed"],
        threshold_parties=CONFIG["threshold_parties"],
        threshold=CONFIG["threshold"],
        use_threshold=True,
    )
    uses_threshold = getattr(he, "threshold_engine", None) is not None or isinstance(
        he, FusedSealThresholdHE
    )
    enc_gadget = EncConsistencyGadget(seed=CONFIG["seed"] + 11) if hasattr(he, "pk") and "p0" in getattr(he, "pk", {}) else None
    if isinstance(he, FusedSealThresholdHE):
        enc_gadget = EncConsistencyGadget(seed=CONFIG["seed"] + 11)
        print(
            f"Fused HE: threshold ({CONFIG['threshold']},{CONFIG['threshold_parties']}) "
            f"+ SEAL sidecar={'on' if he.seal else 'off'}"
        )
    elif uses_threshold:
        print(f"Threshold BFV ({CONFIG['threshold']},{CONFIG['threshold_parties']})")
    else:
        print("SEAL-only backend (no threshold)")
        enc_gadget = None

    zkp = UnruhNormNIZK(
        n_params, CONFIG["norm_threshold"], reps=CONFIG["unruh_reps"], seed=CONFIG["seed"]
    )
    print(f"Unruh NIZK reps={CONFIG['unruh_reps']}")

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
        "he_backend": getattr(he, "backend", backend),
        "robust_agg": robust,
        "enc_consistency": enc_gadget is not None,
        "threshold": CONFIG["threshold"] if uses_threshold else None,
        "unruh_reps": CONFIG["unruh_reps"],
        "n_params": n_params,
        "full_vector_he": True,
        "single_decryptor": not uses_threshold,
        "seal_consistency_err": [],
    }

    rng = np.random.default_rng(CONFIG["seed"])

    for round_t in range(CONFIG["n_rounds"]):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        accepted_cts = []
        detected = 0
        msg = 0

        for cid in range(CONFIG["n_clients"]):
            local = SimpleMLP(n_features, n_classes, hidden=hidden)
            local.set_weights(gw.copy())
            Xc, yc = parts[cid]
            delta = local_training(
                local, Xc, yc, CONFIG["local_epochs"], CONFIG["local_lr"], CONFIG["batch_size"]
            )
            is_mal = cid == CONFIG["malicious_client_id"] and round_t >= 2
            if is_mal:
                delta = rng.normal(0, CONFIG["malicious_scale"], size=len(delta))

            if enc_gadget is not None and hasattr(he, "encrypt_gradient_with_coins"):
                he_cts, coins, pts, _ = he.encrypt_gradient_with_coins(delta)
                enc_proof = enc_gadget.prove_gradient(
                    he.pk, he_cts, pts, coins, he.bfv.n, HE_Q, HE_DELTA
                )
                enc_ok, _ = enc_gadget.verify_gradient(he.pk, he_cts, enc_proof)
                assoc = bind_associated_data(he_cts, enc_proof)
                proof = zkp.generate_proof(delta, associated_data=assoc)
                ok, _ = zkp.verify_proof(proof, associated_data=assoc)
                ok = bool(ok and enc_ok)
            else:
                he_cts, _ = he.encrypt_gradient(delta)
                proof = zkp.generate_proof(delta, associated_data=he_cts)
                ok, _ = zkp.verify_proof(proof, associated_data=he_cts)

            if not ok:
                detected += 1
                print(f"  round {round_t+1}: reject client {cid} (mal={is_mal})")
                continue
            ct_kem, _, _ = kem.encaps(ek)
            accepted_cts.append(he_cts)
            msg += proof.get("proof_size_bytes", 0) + ct_kem["u"].nbytes + ct_kem["v"].nbytes
            if enc_gadget is not None:
                msg += sum(16 * (len(c["c0"]) + len(c["c1"])) for c in he_cts)
            else:
                msg += n_params * 8

        if accepted_cts:
            # Threshold-open each accepted ciphertext, then robust-aggregate
            opened = []
            for cts in accepted_cts:
                vec, _ = he.decrypt_aggregated(cts, 1)
                opened.append(vec)
            update = robust_aggregate(opened, method=robust, f=CONFIG["robust_f"])
            model.set_weights(gw + update)
            if hasattr(he, "last_seal_consistency_err") and he.last_seal_consistency_err is not None:
                metrics["seal_consistency_err"].append(he.last_seal_consistency_err)

        acc, loss = model.evaluate(X_test, y_test)
        rt = time.perf_counter() - t0
        metrics["accuracies"].append(float(acc))
        metrics["round_times"].append(float(rt))
        metrics["detected"].append(int(detected))
        metrics["msg_kb"].append(msg / 1024)
        print(
            f"Round {round_t+1}: acc={acc:.4f} loss={loss:.4f} "
            f"time={rt:.2f}s detected={detected} msg={msg/1024:.1f}KB robust={robust}"
        )

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "target_protocol_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {path}")
    print(
        f"FINAL acc={metrics['accuracies'][-1]:.4f} "
        f"mean_time={np.mean(metrics['round_times']):.2f}s "
        f"backend={metrics['he_backend']} robust={robust}"
    )
    return metrics


if __name__ == "__main__":
    main()
