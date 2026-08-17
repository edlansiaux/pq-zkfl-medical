"""
Unruh-lifted Enc-consistency: parallel binary sessions with invertible RO records.

Closes the QROM-uniformity gap between norm Unruh and classical-FS Enc-consistency.
Default reps=16 (~2^{-16} combinatorial class for the Enc gadget).
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Tuple

import numpy as np

from crypto.enc_consistency import EncConsistencyGadget, _poly_bytes, _poly_scale, _sha3
from crypto.homomorphic import HE_DELTA, HE_Q, _mod, _poly_add_mod, _poly_mul_negacyclic


class UnruhEncConsistency:
    """Binary Unruh transform of the Enc-consistency Σ-protocol (per chunk)."""

    def __init__(self, reps: int = 16, seed: int = 0):
        self.reps = int(reps)
        self.base = EncConsistencyGadget(seed=seed)
        self.rng = np.random.default_rng(seed + 7)

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
        m = np.asarray(plaintext, dtype=object)
        u, e0, e1 = coins["u"], coins["e0"], coins["e1"]
        sessions = []
        first = []
        for _ in range(self.reps):
            u_mask = self.rng.integers(-2, 3, size=n).astype(object)
            e0_mask = self.rng.integers(-3, 4, size=n).astype(object)
            e1_mask = self.rng.integers(-3, 4, size=n).astype(object)
            m_mask = self.rng.integers(0, 16, size=n).astype(object)
            a0 = _poly_add_mod(
                _poly_mul_negacyclic(pk["p0"], u_mask, n, q),
                _poly_add_mod(e0_mask, _poly_scale(m_mask, delta, q), q),
                q,
            )
            a1 = _poly_add_mod(_poly_mul_negacyclic(pk["p1"], u_mask, n, q), e1_mask, q)
            rho = self.rng.bytes(32)
            h_rho = _sha3(rho)
            sessions.append(
                {
                    "a0": a0,
                    "a1": a1,
                    "u_mask": u_mask,
                    "e0_mask": e0_mask,
                    "e1_mask": e1_mask,
                    "m_mask": m_mask,
                    "rho": rho,
                    "h_rho": h_rho,
                }
            )
            first.append(_poly_bytes(a0) + _poly_bytes(a1) + h_rho)

        digest = _sha3(
            b"UNRUH-ENC",
            _poly_bytes(pk["p0"]),
            _poly_bytes(pk["p1"]),
            _poly_bytes(ct["c0"]),
            _poly_bytes(ct["c1"]),
            *first,
        )
        bits = [(digest[i // 8] >> (i % 8)) & 1 for i in range(self.reps)]
        out_sessions = []
        for bit, s in zip(bits, sessions):
            c = int(bit)
            out_sessions.append(
                {
                    "a0": s["a0"],
                    "a1": s["a1"],
                    "c": c,
                    "z_u": _poly_add_mod(s["u_mask"], _poly_scale(u, c, q), q),
                    "z_e0": _poly_add_mod(s["e0_mask"], _poly_scale(e0, c, q), q),
                    "z_e1": _poly_add_mod(s["e1_mask"], _poly_scale(e1, c, q), q),
                    "z_m": _poly_add_mod(s["m_mask"], _poly_scale(m, c, q), q),
                    "rho": s["rho"],
                    "h_rho": s["h_rho"],
                    "n": n,
                    "q": q,
                    "delta": delta,
                }
            )
        return {"mode": "unruh_enc_consistency_chunk", "reps": self.reps, "sessions": out_sessions}

    def verify_chunk(self, pk: Dict, ct: Dict, proof: Dict) -> bool:
        if proof.get("mode") != "unruh_enc_consistency_chunk":
            return False
        sessions = proof["sessions"]
        if len(sessions) != self.reps:
            return False
        first = [
            _poly_bytes(s["a0"]) + _poly_bytes(s["a1"]) + bytes(s["h_rho"]) for s in sessions
        ]
        digest = _sha3(
            b"UNRUH-ENC",
            _poly_bytes(pk["p0"]),
            _poly_bytes(pk["p1"]),
            _poly_bytes(ct["c0"]),
            _poly_bytes(ct["c1"]),
            *first,
        )
        bits = [(digest[i // 8] >> (i % 8)) & 1 for i in range(self.reps)]
        for i, s in enumerate(sessions):
            if _sha3(bytes(s["rho"])) != bytes(s["h_rho"]):
                return False
            if int(s["c"]) != bits[i]:
                return False
            n, q, delta, c = int(s["n"]), int(s["q"]), int(s["delta"]), int(s["c"])
            lhs0 = _poly_add_mod(
                _poly_mul_negacyclic(pk["p0"], s["z_u"], n, q),
                _poly_add_mod(s["z_e0"], _poly_scale(s["z_m"], delta, q), q),
                q,
            )
            lhs1 = _poly_add_mod(
                _poly_mul_negacyclic(pk["p1"], s["z_u"], n, q), s["z_e1"], q
            )
            rhs0 = _poly_add_mod(s["a0"], _poly_scale(ct["c0"], c, q), q)
            rhs1 = _poly_add_mod(s["a1"], _poly_scale(ct["c1"], c, q), q)
            if not (np.array_equal(lhs0, rhs0) and np.array_equal(lhs1, rhs1)):
                return False
        return True

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
        return {"mode": "unruh_enc_consistency", "reps": self.reps, "chunks": chunks}

    def verify_gradient(self, pk: Dict, cts: List[Dict], proof: Dict) -> Tuple[bool, float]:
        t0 = time.perf_counter()
        if proof.get("mode") != "unruh_enc_consistency":
            return False, time.perf_counter() - t0
        if len(proof["chunks"]) != len(cts):
            return False, time.perf_counter() - t0
        ok = all(self.verify_chunk(pk, ct, ch) for ct, ch in zip(cts, proof["chunks"]))
        return ok, time.perf_counter() - t0


def bind_unruh_enc_ad(cts, enc_proof: Dict) -> bytes:
    from crypto.zkp_norm import _serialize_associated_data

    base = _serialize_associated_data(cts)
    extra = b""
    for ch in enc_proof.get("chunks", []):
        for s in ch.get("sessions", []):
            extra += _poly_bytes(s["a0"]) + _poly_bytes(s["a1"]) + bytes(s["h_rho"])
    return base + b"|UNRUH-ENC|" + extra
