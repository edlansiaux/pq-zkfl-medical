"""
Microsoft SEAL backend via TenSEAL (optional production HE path).

Activate with:
  set ZKFL_HE_BACKEND=tenseal
  pip install tenseal

Uses SEAL BFV with HomomorphicEncryption.org-style poly modulus degrees.
Threshold partial-decrypt remains on the NumPy research path; the SEAL path
provides a certified encoder/decryptor for the lattice HE layer.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import tenseal as ts

    HAS_TENSEAL = True
except ImportError:  # pragma: no cover
    ts = None  # type: ignore
    HAS_TENSEAL = False


def seal_available() -> bool:
    return HAS_TENSEAL


class TenSEALGradientHE:
    """Full-vector BFV via Microsoft SEAL (TenSEAL bindings)."""

    def __init__(
        self,
        gradient_dim: int,
        scale: float = 100.0,
        seed: int = 42,
        poly_modulus_degree: Optional[int] = None,
    ):
        if not HAS_TENSEAL:
            raise ImportError("tenseal not installed; pip install tenseal")
        self.gradient_dim = gradient_dim
        self.scale = scale
        # Classic-128 class: 4096; demo: 8192 for headroom or 4096
        preset = os.environ.get("ZKFL_HE_PRESET", "classic128_demo")
        if poly_modulus_degree is None:
            poly_modulus_degree = 4096 if preset == "classic128" else 8192
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.n_chunks = (gradient_dim + self.poly_modulus_degree - 1) // self.poly_modulus_degree

        # SEAL BFV context — plain_modulus must be prime congruent to 1 mod 2N for batching;
        # we use coefficient encoding (no batching) with a large enough t.
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=self.poly_modulus_degree,
            plain_modulus=1032193,  # prime used by TenSEAL examples
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        self.backend = "tenseal-seal"
        self.use_threshold = False
        self.sk = "seal-secret-held-in-context"  # single decryptor; SEAL-certified path
        self.keygen_time = 0.0
        self.n = self.poly_modulus_degree
        # Compatibility shims for callers expecting .bfv.n / .pk
        self.bfv = type("BFVShim", (), {"n": self.n, "q": None, "t": 1032193})()
        self.pk = {"backend": "tenseal", "poly_modulus_degree": self.poly_modulus_degree}

    def encrypt_gradient(self, gradient: np.ndarray) -> Tuple[List, float]:
        t0 = time.perf_counter()
        cts = []
        n = self.poly_modulus_degree
        g = np.asarray(gradient, dtype=np.float64)
        for i in range(self.n_chunks):
            start = i * n
            end = min(start + n, self.gradient_dim)
            chunk = np.zeros(n, dtype=np.int64)
            q = np.round(g[start:end] * self.scale).astype(np.int64)
            chunk[: end - start] = q
            # TenSEAL BFVVector expects list[int]
            cts.append(ts.bfv_vector(self.context, chunk.tolist()))
        return cts, time.perf_counter() - t0

    def aggregate_encrypted_gradients(self, all_ciphertexts: List[List]) -> Tuple[List, float]:
        t0 = time.perf_counter()
        aggregated = []
        for chunk_idx in range(self.n_chunks):
            acc = all_ciphertexts[0][chunk_idx]
            for client in range(1, len(all_ciphertexts)):
                acc = acc + all_ciphertexts[client][chunk_idx]
            aggregated.append(acc)
        return aggregated, time.perf_counter() - t0

    def decrypt_aggregated(self, aggregated_cts: List, n_clients: int) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        result = np.zeros(self.gradient_dim)
        n = self.poly_modulus_degree
        for i, ct in enumerate(aggregated_cts):
            pt = np.array(ct.decrypt(), dtype=np.float64)
            start = i * n
            end = min(start + n, self.gradient_dim)
            result[start:end] = pt[: end - start] / (self.scale * n_clients)
        return result, time.perf_counter() - t0


def create_he_manager(gradient_dim: int, **kwargs):
    """Factory: ZKFL_HE_BACKEND=tenseal|numpy (default numpy)."""
    backend = os.environ.get("ZKFL_HE_BACKEND", "numpy").lower()
    if backend in ("tenseal", "seal"):
        return TenSEALGradientHE(gradient_dim, **{k: v for k, v in kwargs.items() if k in ("scale", "seed", "poly_modulus_degree")})
    from crypto.homomorphic import GradientHEManager

    return GradientHEManager(gradient_dim, **kwargs)
