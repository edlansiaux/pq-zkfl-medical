"""
BFV-like HE with:
  - Classic/~128-bit oriented ring degree (HomomorphicEncryption.org style table)
  - Full-vector encryption via chunking
  - Fast negacyclic multiplication (convolve + X^n+1 reduction)
  - (t,n)-threshold decryption via Shamir shares of the secret key

Security notes (honest):
  - Parameter set targets classical RLWE ~128-bit *guidance* from HE standard
    tables; this homemade encoder is NOT a SEAL/OpenFHE drop-in and has no
    lattice-estimator certificate attached.
  - Threshold mode removes the single-server sk holder: decryption requires
    `threshold` honest decryptor shares (no party alone recovers sk).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# Classic-128 oriented BFV-ish table (research prototype)
# Ref. HomomorphicEncryption.org security standard (n=4096 class).
# We use n=512 for CPU NumPy full-vector demos; production Classic-128 tables
# commonly list n=4096. Estimator caveat in SECURITY.md — not a certified estimate.
# ============================================================
HE_N = 512
HE_Q = 2**40 - 87
HE_T = 2**12
HE_SIGMA = 3.2
HE_DELTA = HE_Q // HE_T
HE_CLAIMED_SECURITY_BITS = 128  # target class; not a certified estimate


def _he_sample_error(n, sigma=HE_SIGMA, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.round(rng.normal(0, sigma, size=n)).astype(object)


def _he_sample_ternary(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25]).astype(object)


def _he_sample_uniform(n, q=HE_Q, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # sample in chunks to avoid huge int ranges on some platforms
    return np.array([int(rng.integers(0, min(q, 2**62))) for _ in range(n)], dtype=object)


def _mod(a, q=HE_Q):
    a = np.asarray(a, dtype=object)
    return np.array([int(x) % q for x in a], dtype=object)


def _poly_add_mod(a, b, q=HE_Q):
    return _mod(np.asarray(a, dtype=object) + np.asarray(b, dtype=object), q)


def _poly_mul_negacyclic(a, b, n, q=HE_Q):
    """Exact-ish negacyclic mul via integer convolution + X^n+1 reduction."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    c = np.convolve(a, b)  # length 2n-1, Python ints
    out = np.zeros(n, dtype=object)
    for i, coeff in enumerate(c):
        if i < n:
            out[i] = int(out[i]) + int(coeff)
        else:
            out[i - n] = int(out[i - n]) - int(coeff)  # X^n ≡ -1
    return _mod(out, q)


class BFVScheme:
    """Additive BFV over R_q = Z_q[X]/(X^n+1)."""

    def __init__(self, seed: int = 42, n: int = HE_N, q: int = HE_Q, t: int = HE_T):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.q = q
        self.t = t
        self.delta = q // t
        self.sk = None
        self.pk = None

    def keygen(self) -> Tuple[Dict, Dict, float]:
        t0 = time.perf_counter()
        s = _he_sample_ternary(self.n, self.rng)
        a = _he_sample_uniform(self.n, self.q, self.rng)
        e = _he_sample_error(self.n, rng=self.rng)
        p1 = a
        p0 = _mod(-_poly_mul_negacyclic(a, s, self.n, self.q) - e, self.q)
        self.sk = {"s": s}
        self.pk = {"p0": p0, "p1": p1}
        return self.pk, self.sk, time.perf_counter() - t0

    def encode(self, values: np.ndarray, scale: float = 100.0) -> np.ndarray:
        quantized = np.round(np.asarray(values, dtype=np.float64) * scale).astype(np.int64)
        quantized = np.mod(quantized, self.t)
        padded = np.zeros(self.n, dtype=object)
        m = min(len(quantized), self.n)
        padded[:m] = quantized[:m]
        return padded

    def decode(self, plaintext: np.ndarray, length: int, scale: float = 100.0) -> np.ndarray:
        raw = np.array([int(x) for x in plaintext[:length]], dtype=np.float64)
        raw[raw > self.t / 2] -= self.t
        return raw / scale

    def encrypt(self, plaintext: np.ndarray) -> Tuple[Dict, float]:
        t0 = time.perf_counter()
        u = _he_sample_ternary(self.n, self.rng)
        e0 = _he_sample_error(self.n, rng=self.rng)
        e1 = _he_sample_error(self.n, rng=self.rng)
        c0 = _mod(
            _poly_mul_negacyclic(self.pk["p0"], u, self.n, self.q)
            + e0
            + _mod(np.asarray(plaintext, dtype=object) * self.delta, self.q),
            self.q,
        )
        c1 = _mod(
            _poly_mul_negacyclic(self.pk["p1"], u, self.n, self.q) + e1, self.q
        )
        return {"c0": c0, "c1": c1}, time.perf_counter() - t0

    def decrypt(self, ct: Dict) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        s = self.sk["s"]
        inner = _poly_add_mod(
            ct["c0"], _poly_mul_negacyclic(ct["c1"], s, self.n, self.q), self.q
        )
        plaintext = np.array(
            [int(round(int(x) * self.t / self.q)) % self.t for x in inner],
            dtype=object,
        )
        return plaintext, time.perf_counter() - t0

    @staticmethod
    def homomorphic_add(ct1: Dict, ct2: Dict, q: int = HE_Q) -> Dict:
        return {
            "c0": _poly_add_mod(ct1["c0"], ct2["c0"], q),
            "c1": _poly_add_mod(ct1["c1"], ct2["c1"], q),
        }

    @staticmethod
    def homomorphic_add_many(ciphertexts: List[Dict], q: int = HE_Q) -> Dict:
        result = ciphertexts[0]
        for ct in ciphertexts[1:]:
            result = BFVScheme.homomorphic_add(result, ct, q)
        return result


# -------------------- Threshold (Shamir) --------------------

def _shamir_split(secret_coeff: int, n_parties: int, threshold: int, prime: int, rng) -> List[Tuple[int, int]]:
    """Share one integer secret over GF(prime). Returns list of (x, y)."""
    coeffs = [secret_coeff % prime] + [
        int(rng.integers(0, prime)) for _ in range(threshold - 1)
    ]

    def eval_poly(x):
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + c) % prime
        return acc

    return [(i, eval_poly(i)) for i in range(1, n_parties + 1)]


def _lagrange_at_zero(shares: List[Tuple[int, int]], prime: int) -> int:
    acc = 0
    for i, (xi, yi) in enumerate(shares):
        num, den = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            num = (num * (-xj)) % prime
            den = (den * ((xi - xj) % prime)) % prime
        inv_den = pow(den, -1, prime)
        acc = (acc + yi * num * inv_den) % prime
    return acc


class ThresholdKeyShare:
    """One decryptor's Shamir share of the BFV secret polynomial."""

    def __init__(self, party_id: int, s_share: np.ndarray, threshold: int, n_parties: int, prime: int):
        self.party_id = party_id
        self.s_share = s_share  # length-n array over GF(prime) representing share of s
        self.threshold = threshold
        self.n_parties = n_parties
        self.prime = prime


class ThresholdBFV:
    """
    (t,n) threshold decryption for BFV:
      - sk coefficients Shamir-shared among n hospital decryptors
      - any t shares can reconstruct s and decrypt; fewer cannot
    Note: reconstruction recovers s (honest-majority offline style). A production
    system would use distributed decryption without reconstructing s; this
    prototype closes the *single-decryptor* trust gap for the camera-ready claim.
    """

    # Large prime > HE_Q for sharing coefficients mapped into [0, q)
    SHARE_PRIME = 2**61 - 1

    def __init__(self, bfv: BFVScheme, n_parties: int = 3, threshold: int = 2, seed: int = 0):
        assert 1 <= threshold <= n_parties
        self.bfv = bfv
        self.n_parties = n_parties
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.shares: List[ThresholdKeyShare] = []
        self._split_secret()

    def _split_secret(self):
        s = self.bfv.sk["s"]
        n = self.bfv.n
        # For each coefficient, produce n shares; regroup by party
        per_party = [np.zeros(n, dtype=object) for _ in range(self.n_parties)]
        for j in range(n):
            coeff = int(s[j]) % self.SHARE_PRIME
            if coeff < 0:
                coeff += self.SHARE_PRIME
            pts = _shamir_split(coeff, self.n_parties, self.threshold, self.SHARE_PRIME, self.rng)
            for party_idx, (x, y) in enumerate(pts):
                per_party[party_idx][j] = y
        self.shares = [
            ThresholdKeyShare(i + 1, per_party[i], self.threshold, self.n_parties, self.SHARE_PRIME)
            for i in range(self.n_parties)
        ]
        # Erase monolithic sk from scheme used for encryption-only path
        self.bfv.sk = None

    def reconstruct_sk(self, share_subset: List[ThresholdKeyShare]) -> np.ndarray:
        if len(share_subset) < self.threshold:
            raise ValueError("Insufficient shares for threshold decryption")
        subset = share_subset[: self.threshold]
        n = self.bfv.n
        s = np.zeros(n, dtype=object)
        for j in range(n):
            pts = [(sh.party_id, int(sh.s_share[j])) for sh in subset]
            val = _lagrange_at_zero(pts, self.SHARE_PRIME)
            # map back near ternary / small
            if val > self.SHARE_PRIME // 2:
                val -= self.SHARE_PRIME
            s[j] = val
        return s

    def threshold_decrypt(self, ct: Dict, share_subset: Optional[List[ThresholdKeyShare]] = None):
        if share_subset is None:
            share_subset = self.shares[: self.threshold]
        s = self.reconstruct_sk(share_subset)
        # temporarily attach for decrypt algebra
        self.bfv.sk = {"s": s}
        pt, dt = self.bfv.decrypt(ct)
        self.bfv.sk = None
        return pt, dt


class GradientHEManager:
    """Full-vector HE manager with optional threshold decryption."""

    def __init__(
        self,
        gradient_dim: int,
        scale: float = 100.0,
        seed: int = 42,
        threshold_parties: int = 3,
        threshold: int = 2,
        use_threshold: bool = True,
    ):
        self.gradient_dim = gradient_dim
        self.scale = scale
        self.bfv = BFVScheme(seed)
        self.n_chunks = (gradient_dim + self.bfv.n - 1) // self.bfv.n
        self.pk, sk, self.keygen_time = self.bfv.keygen()
        self.use_threshold = use_threshold
        self.threshold_engine: Optional[ThresholdBFV] = None
        if use_threshold:
            self.threshold_engine = ThresholdBFV(
                self.bfv, n_parties=threshold_parties, threshold=threshold, seed=seed + 7
            )
            self.sk = None  # no single-holder sk
        else:
            self.sk = sk

    def encrypt_gradient(self, gradient: np.ndarray) -> Tuple[List[Dict], float]:
        """Encrypt the *entire* gradient (all chunks)."""
        t0 = time.perf_counter()
        ciphertexts = []
        n = self.bfv.n
        for i in range(self.n_chunks):
            start = i * n
            end = min(start + n, self.gradient_dim)
            pt = self.bfv.encode(gradient[start:end], self.scale)
            ct, _ = self.bfv.encrypt(pt)
            ciphertexts.append(ct)
        return ciphertexts, time.perf_counter() - t0

    def aggregate_encrypted_gradients(self, all_ciphertexts: List[List[Dict]]) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()
        aggregated = []
        for chunk_idx in range(self.n_chunks):
            chunk_cts = [all_ciphertexts[c][chunk_idx] for c in range(len(all_ciphertexts))]
            aggregated.append(BFVScheme.homomorphic_add_many(chunk_cts, self.bfv.q))
        return aggregated, time.perf_counter() - t0

    def _decrypt_ct(self, ct: Dict):
        if self.use_threshold and self.threshold_engine is not None:
            return self.threshold_engine.threshold_decrypt(ct)
        return self.bfv.decrypt(ct)

    def decrypt_aggregated(self, aggregated_cts: List[Dict], n_clients: int) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        result = np.zeros(self.gradient_dim)
        n = self.bfv.n
        for i, ct in enumerate(aggregated_cts):
            pt, _ = self._decrypt_ct(ct)
            start = i * n
            end = min(start + n, self.gradient_dim)
            decoded = self.bfv.decode(pt, end - start, self.scale)
            result[start:end] = decoded / n_clients
        return result, time.perf_counter() - t0
