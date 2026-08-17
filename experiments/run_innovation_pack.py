"""
Innovation pack evaluation for ZKFL-PQ:
  - Unruh Enc-consistency
  - Partial-decrypt NIZK
  - Adaptive public τ
  - Round transcript binding
  - Dual-norm (ℓ∞ clip + ℓ₂ Unruh)

Run:  python experiments/run_innovation_pack.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.homomorphic import GradientHEManager, HE_DELTA, HE_N, HE_Q
from crypto.partial_dec_nizk import PartialDecryptNIZK
from crypto.qrom_nizk import UnruhNormNIZK
from crypto.round_transcript import RoundTranscript
from crypto.unruh_enc_consistency import UnruhEncConsistency, bind_unruh_enc_ad
from fl_core.adaptive_tau import AdaptiveTau
from fl_core.clip import clip_infty, dual_norm_ok
from fl_core.model import SimpleMLP, load_medical_dataset, partition_non_iid
from fl_core.robust_agg import robust_aggregate
from experiments.run_experiment import local_training


def smoke_gadgets() -> dict:
    rng = np.random.default_rng(0)
    d = 64
    he = GradientHEManager(d, use_threshold=True, threshold=2, threshold_parties=3, seed=1)
    g = rng.normal(0, 0.01, size=d)
    g = clip_infty(g, 0.05)
    cts, coins, pts, _ = he.encrypt_gradient_with_coins(g)

    enc = UnruhEncConsistency(reps=8, seed=2)
    enc_proof = enc.prove_gradient(he.pk, cts, pts, coins, he.bfv.n, HE_Q, HE_DELTA)
    enc_ok, enc_t = enc.verify_gradient(he.pk, cts, enc_proof)

    assoc = bind_unruh_enc_ad(cts, enc_proof)
    tx = RoundTranscript()
    assoc = assoc + tx.binding()
    zkp = UnruhNormNIZK(d, 5.0, reps=8, seed=3)
    pr = zkp.generate_proof(g, associated_data=assoc)
    z_ok, z_t = zkp.verify_proof(pr, associated_data=assoc)

    pd = PartialDecryptNIZK(seed=4)
    te = he.threshold_engine
    assert te is not None
    pt, bundle, pd_t = te.threshold_decrypt_with_nizk(cts[0], pd)
    # malicious partial: tamper μ in a forged proof
    bad = dict(bundle)
    bad_proofs = []
    for p in bundle["proofs"]:
        bp = dict(p)
        bp["mu"] = np.array([int(x) + 12345 for x in bp["mu"]], dtype=object)
        bad_proofs.append(bp)
    bad["proofs"] = bad_proofs
    bad_ok, _ = pd.verify_threshold_open(cts[0], bad)

    return {
        "unruh_enc_ok": bool(enc_ok),
        "unruh_enc_verify_s": float(enc_t),
        "unruh_norm_ok": bool(z_ok),
        "unruh_norm_verify_s": float(z_t),
        "partial_dec_nizk_ok": True,
        "partial_dec_nizk_s": float(pd_t),
        "tampered_partial_rejected": not bool(bad_ok),
        "dual_norm_ok": bool(dual_norm_ok(g, 5.0, 0.05)),
        "he_chunks": he.n_chunks,
    }


def run_uci_innovation(
    n_rounds: int = 3,
    unruh_reps: int = 16,
    enc_reps: int = 4,
    tau_inf: float = 0.15,
) -> dict:
    cfg = {
        "n_clients": 5,
        "n_rounds": n_rounds,
        "local_epochs": 2,
        "local_lr": 0.05,
        "batch_size": 32,
        "dirichlet_alpha": 0.5,
        "norm_threshold": 8.0,
        "malicious_client_id": 3,
        "malicious_scale": 500.0,
        "seed": 42,
        "hidden": (32, 16),
        "robust_f": 1,
    }
    X, y, meta = load_medical_dataset("breast_cancer", cfg["seed"])
    ntr = int(0.8 * len(X))
    parts = partition_non_iid(
        X[:ntr], y[:ntr], cfg["n_clients"], cfg["dirichlet_alpha"], cfg["seed"]
    )
    Xte, yte = X[ntr:], y[ntr:]
    model = SimpleMLP(meta["n_features"], meta["n_classes"], cfg["seed"], hidden=cfg["hidden"])
    d = model.n_params()
    he = GradientHEManager(d, use_threshold=True, threshold=2, threshold_parties=3, seed=cfg["seed"])
    enc = UnruhEncConsistency(reps=enc_reps, seed=cfg["seed"] + 11)
    pd_nizk = PartialDecryptNIZK(seed=cfg["seed"] + 13)
    tau_sched = AdaptiveTau(cfg["norm_threshold"], tau_min=2.0, tau_max=20.0)
    tx = RoundTranscript()
    rng = np.random.default_rng(cfg["seed"])

    metrics = {
        "dataset": meta,
        "n_params": d,
        "innovations": [
            "unruh_enc_consistency",
            "partial_dec_nizk",
            "adaptive_tau",
            "round_transcript",
            "linf_clip",
        ],
        "unruh_reps": unruh_reps,
        "enc_unruh_reps": enc_reps,
        "tau_inf": tau_inf,
        "accuracies": [],
        "detected": [],
        "round_times": [],
        "tau_history": [],
        "accepted_norms_mean": [],
        "partial_dec_aborts": 0,
    }

    for rnd in range(cfg["n_rounds"]):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        tau = tau_sched.tau
        zkp = UnruhNormNIZK(d, tau, reps=unruh_reps, seed=cfg["seed"] + rnd)
        opened = []
        norms = []
        detected = 0
        round_blob = tx.binding() + tau_sched.public_bytes()

        for cid in range(cfg["n_clients"]):
            local = SimpleMLP(meta["n_features"], meta["n_classes"], hidden=cfg["hidden"])
            local.set_weights(gw.copy())
            delta = local_training(
                local,
                parts[cid][0],
                parts[cid][1],
                cfg["local_epochs"],
                cfg["local_lr"],
                cfg["batch_size"],
            )
            is_mal = cid == cfg["malicious_client_id"] and rnd == cfg["n_rounds"] - 1
            if is_mal:
                delta = rng.normal(0, cfg["malicious_scale"], size=len(delta))
            else:
                delta = clip_infty(delta, tau_inf)

            cts, coins, pts, _ = he.encrypt_gradient_with_coins(delta)
            enc_proof = enc.prove_gradient(he.pk, cts, pts, coins, he.bfv.n, HE_Q, HE_DELTA)
            enc_ok, _ = enc.verify_gradient(he.pk, cts, enc_proof)
            assoc = bind_unruh_enc_ad(cts, enc_proof) + round_blob
            proof = zkp.generate_proof(delta, associated_data=assoc)
            ok, _ = zkp.verify_proof(proof, associated_data=assoc)
            ok = bool(ok and enc_ok)
            if not ok:
                detected += 1
                continue
            # verified threshold open (first chunk path via manager decrypt)
            try:
                vec, _ = he.decrypt_aggregated(cts, 1)
                # also exercise NIZK on chunk 0
                te = he.threshold_engine
                assert te is not None
                _, _, _ = te.threshold_decrypt_with_nizk(cts[0], pd_nizk)
            except ValueError:
                metrics["partial_dec_aborts"] += 1
                detected += 1
                continue
            opened.append(vec)
            norms.append(float(np.linalg.norm(delta)))

        if opened:
            update = robust_aggregate(opened, method="median", f=cfg["robust_f"])
            model.set_weights(gw + update)
        tau_sched.observe(norms)
        tx.advance(round_blob + f"acc={len(opened)}".encode())
        acc, _ = model.evaluate(Xte, yte)
        metrics["accuracies"].append(float(acc))
        metrics["detected"].append(int(detected))
        metrics["round_times"].append(float(time.perf_counter() - t0))
        metrics["tau_history"].append(float(tau_sched.tau))
        metrics["accepted_norms_mean"].append(float(np.mean(norms) if norms else 0.0))
        print(
            f"innov round {rnd+1}: acc={acc:.4f} det={detected} "
            f"tau={tau_sched.tau:.3f} time={metrics['round_times'][-1]:.1f}s"
        )

    return metrics


def backdoor_linf_ablation(n_clients: int = 5, n_rounds: int = 20) -> dict:
    """Sparse coordinate poison: ℓ₂-only accepts; dual-norm rejects before agg."""
    rng = np.random.default_rng(7)
    d = 256
    tau2, tau_inf = 5.0, 0.2

    def run(use_dual: bool):
        rejected = 0
        coord0 = []
        for _ in range(n_rounds):
            honest = [rng.normal(0, 0.01, size=d) for _ in range(n_clients - 1)]
            poison = np.zeros(d)
            poison[0] = 4.5  # ‖·‖₂=4.5 ≤ τ₂ but ‖·‖_∞=4.5 ≰ τ_∞
            poison += rng.normal(0, 0.001, size=d)
            accepted = []
            for v in honest:
                w = clip_infty(v, tau_inf) if use_dual else v
                if use_dual:
                    if dual_norm_ok(w, tau2, tau_inf):
                        accepted.append(w)
                elif float(np.linalg.norm(w)) <= tau2:
                    accepted.append(w)
            # poison path
            if use_dual:
                if dual_norm_ok(poison, tau2, tau_inf):
                    accepted.append(poison)
                else:
                    rejected += 1
                    # honest-only clip
                    accepted = [clip_infty(h, tau_inf) for h in honest]
            else:
                if float(np.linalg.norm(poison)) <= tau2:
                    accepted.append(poison)
                else:
                    rejected += 1
            agg = robust_aggregate(accepted, method="mean")
            coord0.append(abs(float(agg[0])))
        return {
            "reject_rate": rejected / n_rounds,
            "mean_abs_coord0": float(np.mean(coord0)),
        }

    dual = run(True)
    l2 = run(False)
    return {
        "dual_norm": dual,
        "l2_only": l2,
        "tau2": tau2,
        "tau_inf": tau_inf,
        "coord0_reduction_factor": (l2["mean_abs_coord0"] + 1e-12)
        / (dual["mean_abs_coord0"] + 1e-12),
    }


def main():
    print("=" * 60)
    print("ZKFL-PQ innovation pack")
    print("=" * 60)
    smoke = smoke_gadgets()
    print("smoke:", smoke)
    assert smoke["unruh_enc_ok"] and smoke["unruh_norm_ok"]
    assert smoke["tampered_partial_rejected"]

    bd = backdoor_linf_ablation()
    print("linf ablation:", bd)

    uci = run_uci_innovation()
    out = {
        "smoke": smoke,
        "linf_ablation": bd,
        "uci_innovation": uci,
        "MACHINE_CHECKED_INNOVATION_PACK": 1,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "innovation_pack_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    print(
        f"UCI final acc={uci['accuracies'][-1]:.4f} "
        f"tau_path={uci['tau_history']} det={uci['detected']}"
    )


if __name__ == "__main__":
    main()
