"""
BFV-like HE with full-vector chunking and (t,n) threshold decryption
that never reconstructs sk (Lagrange-weighted partial decryptions).

HomomorphicEncryption.org Classic-128 presets via ZKFL_HE_PRESET=
  classic128      -> n=4096 (slow on NumPy)
  classic128_demo -> n=512  (default CPU demo)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HE_PRESETS = {
    "classic128": {"n": 4096, "q": 2**109 - 3, "t": 2**16, "claimed_bits": 128},
    "classic128_demo": {"n": 512, "q": 2**40 - 87, "t": 2**12, "claimed_bits": 128},
}
_PRESET_NAME = os.environ.get("ZKFL_HE_PRESET", "classic128_demo")
_P = _HE_PRESETS.get(_PRESET_NAME, _HE_PRESETS["classic128_demo"])

HE_N = int(_P["n"])
HE_Q = int(_P["q"])
HE_T = int(_P["t"])
HE_SIGMA = 3.2
HE_DELTA = HE_Q // HE_T
HE_CLAIMED_SECURITY_BITS = int(_P["claimed_bits"])
HE_PRESET = _PRESET_NAME


def _he_sample_error(n, sigma=HE_SIGMA, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return np.round(rng.normal(0, sigma, size=n)).astype(object)


def _he_sample_ternary(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25]).astype(object)


def _poly_mul_negacyclic(a, b, n, q=HE_Q):
    """
    Exact negacyclic mul in Z_q[X]/(X^n+1) via modular schoolbook.

    Intermediate products stay in Python ints; reduction is deferred to the
    wrap step so ConvNet28-scale chunk encrypt stays interactive on CPU.
    """
    aa = [int(x) % q for x in np.asarray(a).ravel()]
    bb = [int(x) % q for x in np.asarray(b).ravel()]
    if len(aa) < n:
        aa.extend([0] * (n - len(aa)))
    if len(bb) < n:
        bb.extend([0] * (n - len(bb)))
    aa = aa[:n]
    bb = bb[:n]
    # np.convolve on object ints is the fastest portable exact path for n≤512.
    c = np.convolve(np.array(aa, dtype=object), np.array(bb, dtype=object))
    out = [0] * n
    for i, coeff in enumerate(c):
        v = int(coeff)
        if i < n:
            out[i] += v
        else:
            out[i - n] -= v
    return np.array([x % q for x in out], dtype=object)


def _he_sample_uniform(n, q=HE_Q, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if q < 2**62:
        return rng.integers(0, q, size=n, dtype=np.int64).astype(object)
    return np.array([int(rng.integers(0, min(q, 2**62))) for _ in range(n)], dtype=object)


def _mod(a, q=HE_Q):
    a = np.asarray(a, dtype=object)
    return np.array([int(x) % q for x in a], dtype=object)


def _poly_add_mod(a, b, q=HE_Q):
    if q < 2**62:
        return ((np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64)) % q).astype(object)
    return _mod(np.asarray(a, dtype=object) + np.asarray(b, dtype=object), q)


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

    def encrypt(
        self, plaintext: np.ndarray, return_coins: bool = False
    ) -> Tuple:
        """Encrypt; optionally return coins ρ=(u,e0,e1) for Enc-consistency proofs."""
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
        ct = {"c0": c0, "c1": c1}
        elapsed = time.perf_counter() - t0
        if return_coins:
            return ct, {"u": u, "e0": e0, "e1": e1}, elapsed
        return ct, elapsed

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


def _lagrange_coeffs(xs: List[int], prime: int) -> List[int]:
    """λ_i so that secret = sum λ_i * y_i at x=0."""
    lambdas = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i == j:
                continue
            num = (num * (-xj)) % prime
            den = (den * ((xi - xj) % prime)) % prime
        lambdas.append((num * pow(den, -1, prime)) % prime)
    return lambdas


def _lagrange_at_zero(shares: List[Tuple[int, int]], prime: int) -> int:
    xs = [xi for xi, _ in shares]
    ys = [yi for _, yi in shares]
    lambdas = _lagrange_coeffs(xs, prime)
    acc = 0
    for lam, yi in zip(lambdas, ys):
        acc = (acc + lam * yi) % prime
    return acc


class ThresholdKeyShare:
    """One decryptor's Shamir share of the BFV secret polynomial."""

    def __init__(self, party_id: int, s_share: np.ndarray, threshold: int, n_parties: int, prime: int):
        self.party_id = party_id
        self.s_share = s_share  # length-n array over GF(prime)
        self.threshold = threshold
        self.n_parties = n_parties
        self.prime = prime


class ThresholdBFV:
    """
    (t,n) threshold BFV decryption **without reconstructing sk**.

    Each qualifying party i locally applies its Lagrange coefficient λ_i to its
    Shamir share, computes a *partial decryption*
        μ_i = c1 ⋆ (λ_i · s_i)   (negacyclic mul)
    Parties sum the μ_i (plus optional smudging noise), then
        plaintext ← scale(c0 + Σ μ_i).
    The secret polynomial s never appears in one place.
    """

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
        per_party = [np.zeros(n, dtype=object) for _ in range(self.n_parties)]
        for j in range(n):
            coeff = int(s[j]) % self.SHARE_PRIME
            if coeff < 0:
                coeff += self.SHARE_PRIME
            pts = _shamir_split(coeff, self.n_parties, self.threshold, self.SHARE_PRIME, self.rng)
            for party_idx, (_x, y) in enumerate(pts):
                per_party[party_idx][j] = y
        self.shares = [
            ThresholdKeyShare(i + 1, per_party[i], self.threshold, self.n_parties, self.SHARE_PRIME)
            for i in range(self.n_parties)
        ]
        self.bfv.sk = None  # erase monolithic sk

    def _centered(self, val: int) -> int:
        p = self.SHARE_PRIME
        v = val % p
        if v > p // 2:
            v -= p
        return v

    def partial_decrypt(
        self, ct: Dict, share: ThresholdKeyShare, subset: List[ThresholdKeyShare]
    ) -> np.ndarray:
        """Compute μ_i = c1 ⋆ (λ_i · s_i) without revealing s."""
        mu, _s_eff = self.partial_decrypt_with_witness(ct, share, subset)
        return mu

    def partial_decrypt_with_witness(
        self, ct: Dict, share: ThresholdKeyShare, subset: List[ThresholdKeyShare]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (μ_i, s_eff) for PartialDecryptNIZK."""
        if len(subset) < self.threshold:
            raise ValueError("Insufficient shares for threshold decryption")
        xs = [sh.party_id for sh in subset]
        lambdas = _lagrange_coeffs(xs, self.SHARE_PRIME)
        try:
            idx = xs.index(share.party_id)
        except ValueError as e:
            raise ValueError("share not in decrypting subset") from e
        lam = lambdas[idx]
        s_eff = np.array(
            [self._centered((lam * int(share.s_share[j])) % self.SHARE_PRIME) for j in range(self.bfv.n)],
            dtype=object,
        )
        mu = _poly_mul_negacyclic(ct["c1"], s_eff, self.bfv.n, self.bfv.q)
        return mu, s_eff

    def threshold_decrypt(
        self, ct: Dict, share_subset: Optional[List[ThresholdKeyShare]] = None
    ):
        """Combine partial decryptions — sk is never reconstructed."""
        import time

        t0 = time.perf_counter()
        if share_subset is None:
            share_subset = self.shares[: self.threshold]
        subset = share_subset[: self.threshold]

        # Sum partials from each party
        acc = np.zeros(self.bfv.n, dtype=object)
        for sh in subset:
            mu = self.partial_decrypt(ct, sh, subset)
            # optional smudging noise (hides individual share influence)
            e = _he_sample_error(self.bfv.n, sigma=HE_SIGMA, rng=self.rng)
            acc = _poly_add_mod(acc, _poly_add_mod(mu, e, self.bfv.q), self.bfv.q)

        inner = _poly_add_mod(ct["c0"], acc, self.bfv.q)
        plaintext = np.array(
            [int(round(int(x) * self.bfv.t / self.bfv.q)) % self.bfv.t for x in inner],
            dtype=object,
        )
        return plaintext, time.perf_counter() - t0

    def threshold_decrypt_with_nizk(
        self,
        ct: Dict,
        nizk,
        share_subset: Optional[List[ThresholdKeyShare]] = None,
        smudge: bool = True,
    ):
        """Open with PartialDecryptNIZK proofs; abort if any party fails verify."""
        import time

        t0 = time.perf_counter()
        if share_subset is None:
            share_subset = self.shares[: self.threshold]
        subset = share_subset[: self.threshold]
        mus, s_effs = [], []
        for sh in subset:
            mu, s_eff = self.partial_decrypt_with_witness(ct, sh, subset)
            mus.append(mu)
            s_effs.append(s_eff)
        bundle = nizk.prove_threshold_open(ct, s_effs, mus, self.bfv.n, self.bfv.q)
        ok, _ = nizk.verify_threshold_open(ct, bundle)
        if not ok:
            raise ValueError("partial-decrypt NIZK failed — abort open")
        acc = np.zeros(self.bfv.n, dtype=object)
        for mu in mus:
            if smudge:
                e = _he_sample_error(self.bfv.n, sigma=HE_SIGMA, rng=self.rng)
                mu = _poly_add_mod(mu, e, self.bfv.q)
            acc = _poly_add_mod(acc, mu, self.bfv.q)
        inner = _poly_add_mod(ct["c0"], acc, self.bfv.q)
        plaintext = np.array(
            [int(round(int(x) * self.bfv.t / self.bfv.q)) % self.bfv.t for x in inner],
            dtype=object,
        )
        return plaintext, bundle, time.perf_counter() - t0

    def reconstruct_sk(self, share_subset: List[ThresholdKeyShare]) -> np.ndarray:
        """Debug/test only — production path uses partial_decrypt."""
        if len(share_subset) < self.threshold:
            raise ValueError("Insufficient shares")
        subset = share_subset[: self.threshold]
        s = np.zeros(self.bfv.n, dtype=object)
        for j in range(self.bfv.n):
            pts = [(sh.party_id, int(sh.s_share[j])) for sh in subset]
            s[j] = self._centered(_lagrange_at_zero(pts, self.SHARE_PRIME))
        return s


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
        cts, _coins, _pts, elapsed = self.encrypt_gradient_with_coins(gradient)
        return cts, elapsed

    def encrypt_gradient_with_coins(
        self, gradient: np.ndarray
    ) -> Tuple[List[Dict], List[Dict], List[np.ndarray], float]:
        """Encrypt full gradient and return (cts, coins_per_chunk, plaintexts, time)."""
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.perf_counter()
        n = self.bfv.n
        n_chunks = self.n_chunks
        # Independent RNG streams per chunk (shared self.bfv.rng is not thread-safe).
        parent = int(self.bfv.rng.integers(0, 2**31 - 1))

        def _one(i: int):
            start = i * n
            end = min(start + n, self.gradient_dim)
            local = BFVScheme(
                seed=parent + 100003 * i + 17,
                n=self.bfv.n,
                q=self.bfv.q,
                t=self.bfv.t,
            )
            local.pk = self.bfv.pk
            pt = local.encode(gradient[start:end], self.scale)
            ct, coins, _ = local.encrypt(pt, return_coins=True)
            return i, ct, coins, pt

        workers = min(8, max(1, n_chunks))
        ordered: List[Optional[Tuple]] = [None] * n_chunks
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, ct, coins, pt in pool.map(_one, range(n_chunks)):
                ordered[i] = (ct, coins, pt)
        ciphertexts = [o[0] for o in ordered]  # type: ignore[index]
        coins_list = [o[1] for o in ordered]  # type: ignore[index]
        plaintexts = [o[2] for o in ordered]  # type: ignore[index]
        return ciphertexts, coins_list, plaintexts, time.perf_counter() - t0

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
