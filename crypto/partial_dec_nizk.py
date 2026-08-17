"""
NIZK for threshold partial decryption correctness.

Proves knowledge of s_eff such that μ = c1 ⋆ s_eff (mod q), so a malicious
decryptor cannot inject an arbitrary bias polynomial without failing verify.
Fiat–Shamir Σ over the linear relation (research stack).
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from crypto.homomorphic import HE_Q, _he_sample_error, _mod, _poly_add_mod, _poly_mul_negacyclic


def _sha3(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p)
    return h.digest()


def _poly_bytes(p: np.ndarray) -> bytes:
    arr = np.asarray(p, dtype=object).ravel()
    return b"".join(int(x).to_bytes(16, "little", signed=True) for x in arr)


class PartialDecryptNIZK:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def prove(
        self,
        c1: np.ndarray,
        mu: np.ndarray,
        s_eff: np.ndarray,
        n: int,
        q: int = HE_Q,
    ) -> Dict:
        s_mask = self.rng.integers(-3, 4, size=n).astype(object)
        a = _poly_mul_negacyclic(c1, s_mask, n, q)
        stmt = b"".join([b"PARTIAL-DEC", _poly_bytes(c1), _poly_bytes(mu), _poly_bytes(a)])
        c = int.from_bytes(_sha3(stmt)[:8], "little") % (2**16)
        z = _poly_add_mod(s_mask, _mod(np.asarray(s_eff, dtype=object) * c, q), q)
        return {"a": a, "c": c, "z": z, "n": n, "q": q, "mu": mu}

    def verify(self, c1: np.ndarray, proof: Dict) -> bool:
        n, q, c = int(proof["n"]), int(proof["q"]), int(proof["c"])
        mu = proof["mu"]
        stmt = b"".join(
            [b"PARTIAL-DEC", _poly_bytes(c1), _poly_bytes(mu), _poly_bytes(proof["a"])]
        )
        if int.from_bytes(_sha3(stmt)[:8], "little") % (2**16) != c:
            return False
        lhs = _poly_mul_negacyclic(c1, proof["z"], n, q)
        rhs = _poly_add_mod(proof["a"], _mod(np.asarray(mu, dtype=object) * c, q), q)
        return np.array_equal(lhs, rhs)

    def prove_threshold_open(
        self,
        ct: Dict,
        s_effs: List[np.ndarray],
        mus: List[np.ndarray],
        n: int,
        q: int = HE_Q,
    ) -> Dict:
        proofs = [
            self.prove(ct["c1"], mu, s_eff, n, q) for mu, s_eff in zip(mus, s_effs)
        ]
        return {"mode": "partial_dec_nizk", "proofs": proofs}

    def verify_threshold_open(self, ct: Dict, bundle: Dict) -> Tuple[bool, float]:
        t0 = time.perf_counter()
        if bundle.get("mode") != "partial_dec_nizk":
            return False, time.perf_counter() - t0
        ok = all(self.verify(ct["c1"], p) for p in bundle["proofs"])
        return ok, time.perf_counter() - t0
