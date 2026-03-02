"""
ML-KEM-768 (Module-Lattice-Based Key Encapsulation Mechanism)
Simplified implementation faithful to FIPS 203 mathematical structure.

Security is based on the hardness of the Module-Learning With Errors (MLWE) problem
over the polynomial ring R_q = Z_q[X]/(X^n + 1).

Parameters (ML-KEM-768):
    n = 256 (polynomial degree)
    k = 3   (module rank)
    q = 3329 (modulus)
    eta_1 = 2, eta_2 = 2 (noise distribution parameters)
"""

import numpy as np
import hashlib
import os
import time


# ============================================================
# ML-KEM-768 Parameters (FIPS 203)
# ============================================================
N_POLY = 256       # Polynomial degree
K_MOD = 3          # Module rank for ML-KEM-768
Q_MOD = 3329       # Prime modulus
ETA_1 = 2          # CBD parameter for key generation
ETA_2 = 2          # CBD parameter for encryption


def _centered_binomial_distribution(eta, size, rng=None):
    """
    Sample from the Centered Binomial Distribution CBD_eta.
    For each coefficient: sample 2*eta bits, compute sum(first eta) - sum(last eta).
    Output is in range [-eta, eta].
    """
    if rng is None:
        rng = np.random.default_rng()
    a = rng.integers(0, 2, size=(size, eta)).sum(axis=1)
    b = rng.integers(0, 2, size=(size, eta)).sum(axis=1)
    return (a - b).astype(np.int64)


def _sample_poly_cbd(eta, rng=None):
    """Sample a polynomial from CBD_eta in R_q."""
    return _centered_binomial_distribution(eta, N_POLY, rng) % Q_MOD


def _sample_uniform_poly(rng=None):
    """Sample a polynomial uniformly from R_q."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, Q_MOD, size=N_POLY, dtype=np.int64)


def _poly_mul_ntt_naive(a, b):
    """
    Polynomial multiplication in R_q = Z_q[X]/(X^n + 1).
    Uses schoolbook multiplication with reduction mod (X^n + 1).
    For production, NTT-based multiplication would be used.
    """
    n = len(a)
    result = np.zeros(2 * n - 1, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            result[i + j] = (result[i + j] + int(a[i]) * int(b[j])) % Q_MOD

    # Reduce mod X^n + 1: coefficient of X^(n+i) maps to -coefficient of X^i
    reduced = np.zeros(n, dtype=np.int64)
    for i in range(n):
        reduced[i] = (result[i] - result[n + i] if (n + i) < len(result) else result[i]) % Q_MOD
    return reduced


def _poly_add(a, b):
    """Polynomial addition in R_q."""
    return (a + b) % Q_MOD


def _compress(x, d):
    """Compress: round(2^d / q * x) mod 2^d."""
    return np.round(x * (2**d) / Q_MOD).astype(np.int64) % (2**d)


def _decompress(x, d):
    """Decompress: round(q / 2^d * x)."""
    return np.round(x * Q_MOD / (2**d)).astype(np.int64) % Q_MOD


class MLKEM768:
    """
    ML-KEM-768 Key Encapsulation Mechanism.

    KeyGen() -> (ek, dk)
    Encaps(ek) -> (ciphertext, shared_secret)
    Decaps(dk, ciphertext) -> shared_secret
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _sample_matrix_A(self, seed_bytes):
        """Generate the public matrix A ∈ R_q^{k×k} from a seed."""
        A = np.zeros((K_MOD, K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            for j in range(K_MOD):
                h = hashlib.sha256(seed_bytes + bytes([i, j])).digest()
                rng_ij = np.random.default_rng(int.from_bytes(h[:8], 'little'))
                A[i, j] = rng_ij.integers(0, Q_MOD, size=N_POLY, dtype=np.int64)
        return A

    def keygen(self):
        """
        Generate ML-KEM-768 key pair.

        Returns:
            ek: encapsulation key (public)
            dk: decapsulation key (secret)
        """
        t_start = time.perf_counter()

        # Seed for matrix A
        rho = os.urandom(32)
        A = self._sample_matrix_A(rho)

        # Secret vector s ∈ R_q^k sampled from CBD_eta1
        s = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            s[i] = _sample_poly_cbd(ETA_1, self.rng)

        # Error vector e ∈ R_q^k sampled from CBD_eta1
        e = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            e[i] = _sample_poly_cbd(ETA_1, self.rng)

        # Public key: t = A·s + e (mod q)
        t = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            for j in range(K_MOD):
                t[i] = _poly_add(t[i], _poly_mul_ntt_naive(A[i, j], s[j]))
            t[i] = _poly_add(t[i], e[i])

        t_elapsed = time.perf_counter() - t_start

        ek = {'rho': rho, 'A': A, 't': t}
        dk = {'s': s, 'ek': ek}

        return ek, dk, t_elapsed

    def encaps(self, ek):
        """
        Encapsulate: generate shared secret and ciphertext.

        Args:
            ek: encapsulation key

        Returns:
            ciphertext, shared_secret, time_elapsed
        """
        t_start = time.perf_counter()

        A = ek['A']
        t = ek['t']

        # Random vector r ∈ R_q^k from CBD_eta1
        r = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            r[i] = _sample_poly_cbd(ETA_1, self.rng)

        # Error vectors
        e1 = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            e1[i] = _sample_poly_cbd(ETA_2, self.rng)
        e2 = _sample_poly_cbd(ETA_2, self.rng)

        # Random message for shared secret derivation
        m = self.rng.integers(0, 2, size=N_POLY, dtype=np.int64)

        # u = A^T · r + e1
        u = np.zeros((K_MOD, N_POLY), dtype=np.int64)
        for i in range(K_MOD):
            for j in range(K_MOD):
                u[i] = _poly_add(u[i], _poly_mul_ntt_naive(A[j, i], r[j]))
            u[i] = _poly_add(u[i], e1[i])

        # v = t^T · r + e2 + ⌈q/2⌋ · m
        v = np.copy(e2)
        for i in range(K_MOD):
            v = _poly_add(v, _poly_mul_ntt_naive(t[i], r[i]))
        v = _poly_add(v, (m * (Q_MOD // 2)) % Q_MOD)

        # Derive shared secret via hash
        ct_bytes = u.tobytes() + v.tobytes()
        shared_secret = hashlib.sha256(ct_bytes + m.tobytes()).digest()

        t_elapsed = time.perf_counter() - t_start

        ciphertext = {'u': u, 'v': v, 'm_hash': hashlib.sha256(m.tobytes()).digest()}
        return ciphertext, shared_secret, t_elapsed

    def decaps(self, dk, ciphertext):
        """
        Decapsulate: recover shared secret from ciphertext.

        Args:
            dk: decapsulation key
            ciphertext: ciphertext from encaps

        Returns:
            shared_secret, time_elapsed
        """
        t_start = time.perf_counter()

        s = dk['s']
        u = ciphertext['u']
        v = ciphertext['v']

        # Recover noisy message: v - s^T · u
        s_dot_u = np.zeros(N_POLY, dtype=np.int64)
        for i in range(K_MOD):
            s_dot_u = _poly_add(s_dot_u, _poly_mul_ntt_naive(s[i], u[i]))

        m_noisy = (v - s_dot_u) % Q_MOD

        # Decode: if coefficient is closer to q/2 -> 1, else -> 0
        threshold = Q_MOD // 4
        m_recovered = np.zeros(N_POLY, dtype=np.int64)
        for i in range(N_POLY):
            val = m_noisy[i]
            # Distance to 0 vs distance to q/2
            dist_0 = min(val, Q_MOD - val)
            dist_half = abs(val - Q_MOD // 2)
            m_recovered[i] = 1 if dist_half < dist_0 else 0

        ct_bytes = u.tobytes() + v.tobytes()
        shared_secret = hashlib.sha256(ct_bytes + m_recovered.tobytes()).digest()

        t_elapsed = time.perf_counter() - t_start
        return shared_secret, t_elapsed


def symmetric_encrypt(key_bytes, plaintext_bytes):
    """AES-256-CTR symmetric encryption using the shared secret as key."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(plaintext_bytes) + encryptor.finalize()
    return iv + ct


def symmetric_decrypt(key_bytes, ciphertext_bytes):
    """AES-256-CTR symmetric decryption."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    iv = ciphertext_bytes[:16]
    ct = ciphertext_bytes[16:]
    cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(iv))
    decryptor = cipher.decryptor()
    return decryptor.update(ct) + decryptor.finalize()


# ============================================================
# Benchmarking utilities
# ============================================================
def benchmark_mlkem(n_trials=5):
    """Run ML-KEM-768 benchmarks."""
    kem = MLKEM768()
    keygen_times = []
    encaps_times = []
    decaps_times = []
    correctness = []

    for _ in range(n_trials):
        ek, dk, t_kg = kem.keygen()
        keygen_times.append(t_kg)

        ct, ss_enc, t_enc = kem.encaps(ek)
        encaps_times.append(t_enc)

        ss_dec, t_dec = kem.decaps(dk, ct)
        decaps_times.append(t_dec)

        correctness.append(ss_enc == ss_dec)

    return {
        'keygen_mean': np.mean(keygen_times),
        'encaps_mean': np.mean(encaps_times),
        'decaps_mean': np.mean(decaps_times),
        'correctness_rate': np.mean(correctness),
        'pk_size_bytes': K_MOD * N_POLY * 8 + 32,  # t + rho
        'ct_size_bytes': (K_MOD + 1) * N_POLY * 8,  # u + v
    }
