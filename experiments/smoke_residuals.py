"""Smoke test: Enc-consistency + optional TenSEAL + Unruh combinatorial checker."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_enc_consistency():
    from crypto.enc_consistency import EncConsistencyGadget
    from crypto.homomorphic import BFVScheme, HE_DELTA, HE_Q

    bfv = BFVScheme(seed=1)
    bfv.keygen()
    pt = np.arange(bfv.n) % bfv.t
    ct, coins, _ = bfv.encrypt(pt, return_coins=True)
    g = EncConsistencyGadget(seed=2)
    proof = g.prove_chunk(bfv.pk, ct, pt, coins, bfv.n, HE_Q, HE_DELTA)
    assert g.verify_chunk(bfv.pk, ct, proof)
    # Tamper ciphertext → fail
    bad = {"c0": ct["c0"].copy(), "c1": ct["c1"].copy()}
    bad["c0"][0] = (int(bad["c0"][0]) + 1) % HE_Q
    assert not g.verify_chunk(bfv.pk, bad, proof)
    print("[ok] enc_consistency")


def test_tenseal():
    from crypto.seal_backend import HAS_TENSEAL, TenSEALGradientHE

    if not HAS_TENSEAL:
        print("[skip] tenseal not installed")
        return
    he = TenSEALGradientHE(64, scale=100.0, seed=0, poly_modulus_degree=8192)
    g = np.random.default_rng(0).normal(0, 0.01, size=64)
    cts, _ = he.encrypt_gradient(g)
    agg, _ = he.aggregate_encrypted_gradients([cts, cts])
    mean, _ = he.decrypt_aggregated(agg, 2)
    err = float(np.mean(np.abs(mean - g)))
    print(f"[ok] tenseal err={err:.6f}")
    assert err < 0.05


def test_unruh_formal():
    from formal.check_unruh_soundness import main

    assert main() == 0


if __name__ == "__main__":
    test_enc_consistency()
    test_tenseal()
    test_unruh_formal()
    print("ALL_RESIDUAL_SMOKES_OK")
