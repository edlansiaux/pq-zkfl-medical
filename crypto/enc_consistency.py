"""
Enc-consistency gadget for BFV encryption coins ρ = (u, e0, e1).

Proves knowledge of coins and plaintext chunk m such that
  c0 ≡ p0⋆u + e0 + Δ·m  (mod q)
  c1 ≡ p1⋆u + e1         (mod q)
without relying only on transcript hashing of ciphertext bytes.

Fiat–Shamir Σ-protocol (classical ROM) bound to (pk, ct). Combined with the
norm Unruh/FS proof on the same quantized vector (shared associated_data),
this closes the Enc-consistency residual flagged in the camera-ready.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crypto.homomorphic import HE_DELTA, HE_Q, _mod, _poly_add_mod, _poly_mul_negacyclic


def _sha3(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p)
    return h.digest()


def _poly_bytes(p: np.ndarray) -> bytes:
    arr = np.asarray(p, dtype=object).ravel()
    return b"".join(int(x).to_bytes(16, "little", signed=True) for x in arr)


def _poly_scale(p: np.ndarray, k: int, q: int) -> np.ndarray:
    return _mod(np.asarray(p, dtype=object) * int(k), q)


class EncConsistencyGadget:
    """Dedicated opening gadget for BFV encryption randomness."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def prove_chunk(
        self,
        pk: Dict,
        ct: Dict,
        plaintext: np.ndarray,
        coins: Dict[str, np.ndarray],
        n: int,
        q: int = HE_Q,
        delta: int = HE_DELTA,
    ) -> Dict:
        """Σ-protocol FS proof that (coins, m) encrypt to ct under pk."""
        m = np.asarray(plaintext, dtype=object)
        u, e0, e1 = coins["u"], coins["e0"], coins["e1"]

        # Masks
        u_mask = self.rng.integers(-2, 3, size=n).astype(object)
        e0_mask = self.rng.integers(-3, 4, size=n).astype(object)
        e1_mask = self.rng.integers(-3, 4, size=n).astype(object)
        m_mask = self.rng.integers(0, 16, size=n).astype(object)

        # First message: Enc(pk, m_mask; masks)  (same linear form)
        a0 = _poly_add_mod(
            _poly_mul_negacyclic(pk["p0"], u_mask, n, q),
            _poly_add_mod(e0_mask, _poly_scale(m_mask, delta, q), q),
            q,
        )
        a1 = _poly_add_mod(_poly_mul_negacyclic(pk["p1"], u_mask, n, q), e1_mask, q)

        stmt = b"".join(
            [
                b"ENC-CONSIST",
                _poly_bytes(pk["p0"]),
                _poly_bytes(pk["p1"]),
                _poly_bytes(ct["c0"]),
                _poly_bytes(ct["c1"]),
                _poly_bytes(a0),
                _poly_bytes(a1),
            ]
        )
        challenge = int.from_bytes(_sha3(stmt)[:8], "little") % (2**16)

        # Response
        z_u = _poly_add_mod(u_mask, _poly_scale(u, challenge, q), q)
        z_e0 = _poly_add_mod(e0_mask, _poly_scale(e0, challenge, q), q)
        z_e1 = _poly_add_mod(e1_mask, _poly_scale(e1, challenge, q), q)
        z_m = _poly_add_mod(m_mask, _poly_scale(m, challenge, q), q)

        return {
            "a0": a0,
            "a1": a1,
            "c": challenge,
            "z_u": z_u,
            "z_e0": z_e0,
            "z_e1": z_e1,
            "z_m": z_m,
            "n": n,
            "q": q,
            "delta": delta,
        }

    def verify_chunk(self, pk: Dict, ct: Dict, proof: Dict) -> bool:
        n = int(proof["n"])
        q = int(proof["q"])
        delta = int(proof["delta"])
        c = int(proof["c"])

        stmt = b"".join(
            [
                b"ENC-CONSIST",
                _poly_bytes(pk["p0"]),
                _poly_bytes(pk["p1"]),
                _poly_bytes(ct["c0"]),
                _poly_bytes(ct["c1"]),
                _poly_bytes(proof["a0"]),
                _poly_bytes(proof["a1"]),
            ]
        )
        if int.from_bytes(_sha3(stmt)[:8], "little") % (2**16) != c:
            return False

        # Check: Enc(pk, z_m; z_*) == A + c * ct
        lhs0 = _poly_add_mod(
            _poly_mul_negacyclic(pk["p0"], proof["z_u"], n, q),
            _poly_add_mod(proof["z_e0"], _poly_scale(proof["z_m"], delta, q), q),
            q,
        )
        lhs1 = _poly_add_mod(
            _poly_mul_negacyclic(pk["p1"], proof["z_u"], n, q), proof["z_e1"], q
        )
        rhs0 = _poly_add_mod(proof["a0"], _poly_scale(ct["c0"], c, q), q)
        rhs1 = _poly_add_mod(proof["a1"], _poly_scale(ct["c1"], c, q), q)
        return np.array_equal(lhs0, rhs0) and np.array_equal(lhs1, rhs1)

    def prove_gradient(
        self,
        pk: Dict,
        cts: List[Dict],
        plaintexts: List[np.ndarray],
        coins_list: List[Dict[str, np.ndarray]],
        n: int,
        q: int = HE_Q,
        delta: int = HE_DELTA,
    ) -> Dict:
        chunks = [
            self.prove_chunk(pk, ct, pt, coins, n, q, delta)
            for ct, pt, coins in zip(cts, plaintexts, coins_list)
        ]
        return {"mode": "enc_consistency", "chunks": chunks}

    def verify_gradient(self, pk: Dict, cts: List[Dict], proof: Dict) -> Tuple[bool, float]:
        t0 = time.perf_counter()
        if proof.get("mode") != "enc_consistency":
            return False, time.perf_counter() - t0
        if len(proof["chunks"]) != len(cts):
            return False, time.perf_counter() - t0
        ok = all(
            self.verify_chunk(pk, ct, ch) for ct, ch in zip(cts, proof["chunks"])
        )
        return ok, time.perf_counter() - t0


def bind_associated_data(cts: Any, enc_proof: Optional[Dict] = None) -> bytes:
    """Canonical AD bytes: ciphertext || EncConsistency first messages."""
    from crypto.zkp_norm import _serialize_associated_data

    base = _serialize_associated_data(cts)
    if not enc_proof:
        return base
    extra = b""
    for ch in enc_proof.get("chunks", []):
        extra += _poly_bytes(ch["a0"]) + _poly_bytes(ch["a1"])
    return base + b"|ENC|" + extra
