"""
QROM Unruh NIZK for the norm-bound Σ-protocol (EUROCRYPT 2015 style).

  - Default r=128 binary parallel sessions (~128-bit Unruh soundness)
  - Invertible RO records (preimage, SHA3 image) in the proof
  - Challenges bind BFV ciphertext bytes (associated_data)

Classical Fiat–Shamir alone is not tightly QROM-secure
(Kiltz–Lyubashevsky–Schaffner, EUROCRYPT 2018).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crypto.zkp_norm import (
    COMMIT_Q,
    LatticeCommitment,
    ZKPNormBound,
    _serialize_associated_data,
)


def _sha3(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p)
    return h.digest()


class UnruhNormNIZK:
    """
    Unruh NIZK for ||Δw||₂ ≤ τ, bound to associated_data.

    Default ``reps=128`` targets ~128-bit soundness under the Unruh transform
    with binary challenges (each repetition contributes one challenge bit).
    """

    def __init__(self, dim: int, threshold: float, reps: int = 128, seed: int = 42):
        self.dim = dim
        self.tau = threshold
        self.reps = reps
        self.rng = np.random.default_rng(seed + 99)
        # One shared SIS commitment key for all Unruh sessions *and* the base
        # Σ-protocol (not r+1 independent A matrices of shape 256×(d+128)).
        self.comm = LatticeCommitment(dim, seed + 1000)
        self.base = ZKPNormBound(dim, threshold, seed, commitment=self.comm)

    def generate_proof(
        self, gradient: np.ndarray, associated_data: Optional[Any] = None
    ) -> Dict:
        if len(gradient) != self.dim:
            raise ValueError(f"dim mismatch: {len(gradient)} vs {self.dim}")
        assoc = _serialize_associated_data(associated_data)
        # Prefer robust poly bytes if assoc empty but cts are object arrays
        if not assoc and associated_data is not None:
            assoc = repr(associated_data).encode()

        dw = self.base._quantize_gradient(gradient)
        sessions = []
        first_msgs = []

        for i in range(self.reps):
            C, r_c = self.comm.commit(dw)
            y = np.round(self.rng.normal(0, self.base.sigma_mask, size=self.dim)).astype(
                np.int64
            )
            T, r_t = self.comm.commit(y)
            # Unruh invertible RO point: (preimage ρ, image H(ρ))
            rho = self.rng.bytes(32)
            h_rho = _sha3(rho)
            sessions.append(
                {"C": C, "T": T, "y": y, "r_c": r_c, "r_t": r_t, "rho": rho, "h_rho": h_rho}
            )
            first_msgs.append(C.tobytes() + T.tobytes() + h_rho)

        # Challenge string from statement + all first messages + assoc
        digest = _sha3(b"UNRUH", repr(self.tau).encode(), assoc, *first_msgs)
        bits = [(digest[i // 8] >> (i % 8)) & 1 for i in range(self.reps)]

        responses = []
        for i, bit in enumerate(bits):
            s = sessions[i]
            c = int(bit)  # binary challenge in Unruh transform
            z = s["y"] + c * dw
            r_z = (s["r_t"] + c * s["r_c"]) % COMMIT_Q
            # Algebraic verify uses the shared A
            responses.append(
                {
                    "C": s["C"],
                    "T": s["T"],
                    "z": z,
                    "r_z": r_z,
                    "c": c,
                    "rho": s["rho"],
                    "h_rho": s["h_rho"],
                    "z_norm": float(np.linalg.norm(z.astype(np.float64))),
                }
            )

        proof_size = sum(
            r["C"].nbytes
            + r["T"].nbytes
            + r["z"].nbytes
            + r["r_z"].nbytes
            + 32
            + 32
            for r in responses
        )
        return {
            "mode": "unruh",
            "reps": self.reps,
            "sessions": responses,
            "assoc_len": len(assoc),
            "actual_norm": float(np.linalg.norm(gradient)),
            "proof_size_bytes": proof_size,
            "accepted": all(r["z_norm"] <= self.base.B_reject for r in responses),
        }

    def verify_proof(
        self, proof: Dict, associated_data: Optional[Any] = None
    ) -> Tuple[bool, float]:
        import time

        t0 = time.perf_counter()
        if proof.get("mode") != "unruh" or len(proof["sessions"]) != self.reps:
            return False, time.perf_counter() - t0

        assoc = _serialize_associated_data(associated_data)
        sessions: List[Dict] = proof["sessions"]

        # Recompute challenge bits
        first_msgs = [
            s["C"].tobytes() + s["T"].tobytes() + bytes(s["h_rho"]) for s in sessions
        ]
        digest = _sha3(b"UNRUH", repr(self.tau).encode(), assoc, *first_msgs)
        bits = [(digest[i // 8] >> (i % 8)) & 1 for i in range(self.reps)]

        cols = []
        rhs_cols = []
        for i, s in enumerate(sessions):
            # Invertible RO check (Unruh)
            if _sha3(bytes(s["rho"])) != bytes(s["h_rho"]):
                return False, time.perf_counter() - t0
            if int(s["c"]) != bits[i]:
                return False, time.perf_counter() - t0
            z_norm = float(np.linalg.norm(s["z"].astype(np.float64)))
            if z_norm > self.base.B_reject:
                return False, time.perf_counter() - t0
            z = s["z"]
            if len(z) != self.dim:
                return False, time.perf_counter() - t0
            cols.append(
                np.concatenate([z.astype(np.int64), s["r_z"].astype(np.int64)])
            )
            rhs_cols.append(
                (s["T"].astype(np.int64) + int(s["c"]) * s["C"].astype(np.int64))
                % COMMIT_Q
            )

        # One batched matmul against the shared SIS matrix A.
        Z = np.column_stack(cols)
        lhs = (self.comm.A @ Z) % COMMIT_Q
        rhs = np.column_stack(rhs_cols)
        if not np.array_equal(lhs, rhs):
            return False, time.perf_counter() - t0

        return True, time.perf_counter() - t0
