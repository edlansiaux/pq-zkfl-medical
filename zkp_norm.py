"""
Lattice-Based Homomorphic Encryption for Secure Gradient Aggregation.

BFV-like scheme operating over the polynomial ring R_q = Z_q[X]/(X^n + 1).

The scheme supports additive homomorphism:
    Dec(Enc(m₁) + Enc(m₂)) = m₁ + m₂

Security: Based on the Ring-LWE (RLWE) problem, conjectured quantum-resistant.
"""

import numpy as np
import time
from typing import Tuple, Dict, List


# ============================================================
# BFV-like Parameters (security level ~128 bits)
# ============================================================
HE_N = 512              # Polynomial degree (power of 2)
HE_Q = 2**32 - 5        # Ciphertext modulus (large prime-like)
HE_T = 2**16            # Plaintext modulus
HE_SIGMA = 3.2          # Error standard deviation
HE_DELTA = HE_Q // HE_T # Scaling factor ⌊q/t⌋


def _he_sample_error(n, sigma=HE_SIGMA, rng=None):
    """Sample error polynomial from discrete Gaussian."""
    if rng is None:
        rng = np.random.default_rng()
    return np.round(rng.normal(0, sigma, size=n)).astype(np.int64)


def _he_sample_ternary(n, rng=None):
    """Sample ternary polynomial (coefficients in {-1, 0, 1})."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25]).astype(np.int64)


def _he_sample_uniform(n, q=HE_Q, rng=None):
    """Sample uniform polynomial mod q."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, q, size=n, dtype=np.int64)


def _poly_add_mod(a, b, q=HE_Q):
    """Polynomial addition mod q."""
    return (a.astype(np.int64) + b.astype(np.int64)) % q


def _poly_mul_schoolbook(a, b, n, q=HE_Q):
    """Polynomial multiplication in R_q = Z_q[X]/(X^n + 1)."""
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                result[idx] = (result[idx] + int(a[i]) * int(b[j])) % q
            else:
                result[idx - n] = (result[idx - n] - int(a[i]) * int(b[j])) % q
    return result


class BFVScheme:
    """BFV Homomorphic Encryption Scheme (additive only)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n = HE_N
        self.q = HE_Q
        self.t = HE_T
        self.delta = HE_DELTA
        self.sk = None
        self.pk = None

    def keygen(self) -> Tuple[Dict, Dict, float]:
        """Generate BFV key pair."""
        t_start = time.perf_counter()

        s = _he_sample_ternary(self.n, self.rng)
        a = _he_sample_uniform(self.n, self.q, self.rng)
        e = _he_sample_error(self.n, rng=self.rng)

        p1 = a
        p0 = (-_poly_mul_schoolbook(a, s, self.n, self.q) - e) % self.q

        t_elapsed = time.perf_counter() - t_start

        self.sk = {'s': s}
        self.pk = {'p0': p0, 'p1': p1}
        return self.pk, self.sk, t_elapsed

    def encode(self, values: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Encode a real-valued vector into a plaintext polynomial."""
        quantized = np.round(values * scale).astype(np.int64)
        quantized = quantized % self.t

        if len(quantized) > self.n:
            return quantized[:self.n]
        else:
            padded = np.zeros(self.n, dtype=np.int64)
            padded[:len(quantized)] = quantized
            return padded

    def decode(self, plaintext: np.ndarray, length: int, scale: float = 100.0) -> np.ndarray:
        """Decode a plaintext polynomial back to real-valued vector."""
        raw = plaintext[:length].astype(np.float64)
        raw[raw > self.t / 2] -= self.t
        return raw / scale

    def encrypt(self, plaintext: np.ndarray) -> Tuple[Dict, float]:
        """Encrypt a plaintext polynomial."""
        t_start = time.perf_counter()

        u = _he_sample_ternary(self.n, self.rng)
        e0 = _he_sample_error(self.n, rng=self.rng)
        e1 = _he_sample_error(self.n, rng=self.rng)

        c0 = (_poly_mul_schoolbook(self.pk['p0'], u, self.n, self.q)
              + e0
              + (self.delta * plaintext) % self.q) % self.q
        c1 = (_poly_mul_schoolbook(self.pk['p1'], u, self.n, self.q)
              + e1) % self.q

        t_elapsed = time.perf_counter() - t_start

        ct = {'c0': c0, 'c1': c1}
        return ct, t_elapsed

    def decrypt(self, ct: Dict) -> Tuple[np.ndarray, float]:
        """Decrypt a ciphertext."""
        t_start = time.perf_counter()

        s = self.sk['s']
        inner = (_poly_add_mod(ct['c0'],
                               _poly_mul_schoolbook(ct['c1'], s, self.n, self.q),
                               self.q))

        scaled = (inner.astype(np.float64) * self.t / self.q)
        plaintext = np.round(scaled).astype(np.int64) % self.t

        t_elapsed = time.perf_counter() - t_start
        return plaintext, t_elapsed

    @staticmethod
    def homomorphic_add(ct1: Dict, ct2: Dict) -> Dict:
        """Homomorphic addition of two ciphertexts."""
        return {
            'c0': _poly_add_mod(ct1['c0'], ct2['c0']),
            'c1': _poly_add_mod(ct1['c1'], ct2['c1']),
        }

    @staticmethod
    def homomorphic_add_many(ciphertexts: List[Dict]) -> Dict:
        """Aggregate multiple ciphertexts via homomorphic addition."""
        if len(ciphertexts) == 0:
            raise ValueError("Empty ciphertext list")
        result = ciphertexts[0]
        for ct in ciphertexts[1:]:
            result = BFVScheme.homomorphic_add(result, ct)
        return result


class GradientHEManager:
    """High-level manager for encrypting, aggregating, and decrypting gradients."""

    def __init__(self, gradient_dim: int, scale: float = 100.0, seed: int = 42):
        self.gradient_dim = gradient_dim
        self.scale = scale
        self.bfv = BFVScheme(seed)
        self.n_chunks = (gradient_dim + HE_N - 1) // HE_N

        self.pk, self.sk, self.keygen_time = self.bfv.keygen()

    def encrypt_gradient(self, gradient: np.ndarray) -> Tuple[List[Dict], float]:
        """Encrypt a full gradient vector (with chunking)."""
        t_start = time.perf_counter()
        ciphertexts = []

        for i in range(self.n_chunks):
            start = i * HE_N
            end = min(start + HE_N, self.gradient_dim)
            chunk = gradient[start:end]
            pt = self.bfv.encode(chunk, self.scale)
            ct, _ = self.bfv.encrypt(pt)
            ciphertexts.append(ct)

        t_elapsed = time.perf_counter() - t_start
        return ciphertexts, t_elapsed

    def aggregate_encrypted_gradients(self, all_ciphertexts: List[List[Dict]]) -> Tuple[List[Dict], float]:
        """Aggregate encrypted gradients from multiple clients."""
        t_start = time.perf_counter()
        n_clients = len(all_ciphertexts)
        aggregated = []

        for chunk_idx in range(self.n_chunks):
            chunk_cts = [all_ciphertexts[client_idx][chunk_idx]
                        for client_idx in range(n_clients)]
            agg_ct = BFVScheme.homomorphic_add_many(chunk_cts)
            aggregated.append(agg_ct)

        t_elapsed = time.perf_counter() - t_start
        return aggregated, t_elapsed

    def decrypt_aggregated(self, aggregated_cts: List[Dict], n_clients: int) -> Tuple[np.ndarray, float]:
        """Decrypt aggregated ciphertexts and compute mean."""
        t_start = time.perf_counter()
        result = np.zeros(self.gradient_dim)

        for i, ct in enumerate(aggregated_cts):
            pt, _ = self.bfv.decrypt(ct)
            start = i * HE_N
            end = min(start + HE_N, self.gradient_dim)
            decoded = self.bfv.decode(pt, end - start, self.scale)
            result[start:end] = decoded / n_clients

        t_elapsed = time.perf_counter() - t_start
        return result, t_elapsed


def benchmark_he(dim=1000, n_clients=5, n_trials=3):
    """Benchmark homomorphic encryption operations."""
    results = {
        'keygen_times': [], 'encrypt_times': [], 'aggregate_times': [],
        'decrypt_times': [], 'reconstruction_errors': [], 'ct_sizes': []
    }

    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        manager = GradientHEManager(dim, scale=100.0, seed=trial)
        results['keygen_times'].append(manager.keygen_time)

        gradients = [rng.normal(0, 0.01, size=dim) for _ in range(n_clients)]

        all_cts = []
        for grad in gradients:
            cts, enc_time = manager.encrypt_gradient(grad)
            results['encrypt_times'].append(enc_time)
            all_cts.append(cts)

        agg_cts, agg_time = manager.aggregate_encrypted_gradients(all_cts)
        results['aggregate_times'].append(agg_time)

        decrypted_mean, dec_time = manager.decrypt_aggregated(agg_cts, n_clients)
        results['decrypt_times'].append(dec_time)

        actual_mean = np.mean(gradients, axis=0)
        error = np.mean(np.abs(decrypted_mean - actual_mean))
        results['reconstruction_errors'].append(error)

        ct_size = sum(ct['c0'].nbytes + ct['c1'].nbytes for ct in all_cts[0])
        results['ct_sizes'].append(ct_size)

    return {
        'keygen_time_mean': np.mean(results['keygen_times']),
        'encrypt_time_mean': np.mean(results['encrypt_times']),
        'aggregate_time_mean': np.mean(results['aggregate_times']),
        'decrypt_time_mean': np.mean(results['decrypt_times']),
        'reconstruction_error_mean': np.mean(results['reconstruction_errors']),
        'ct_size_per_client': np.mean(results['ct_sizes']),
    }
