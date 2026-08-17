"""
Zero-Knowledge Proof System for Gradient Norm Bounds in Federated Learning.

Non-interactive Sigma protocol (Fiat-Shamir) proving:
  Statement: ||Δw||₂ ≤ τ  (on the proven vector)
  Binding:   challenge digests associated_data (e.g. BFV ciphertext bytes)

Verification is purely cryptographic:
  1) ||z||₂ ≤ B
  2) Fiat-Shamir challenge consistency (includes associated_data)
  3) Algebraic check A·[z || r_z] ≡ T + c·C (mod q)

The former client-supplied `is_within_bound` flag is NOT used for acceptance.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Lattice commitment parameters
COMMIT_N = 128
COMMIT_Q = 7681
COMMIT_M = 256
REJECTION_BOUND = 12


class LatticeCommitment:
    """SIS-style commitment Commit(m; r) = A·[m || r] mod q."""

    def __init__(self, input_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.randomness_dim = COMMIT_N
        self.rng = np.random.default_rng(seed)
        self.total_cols = input_dim + COMMIT_N
        self.A = self.rng.integers(
            0, COMMIT_Q, size=(COMMIT_M, self.total_cols), dtype=np.int64
        )

    def commit(self, message: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(message) < self.input_dim:
            message = np.pad(message, (0, self.input_dim - len(message)))
        elif len(message) > self.input_dim:
            message = message[: self.input_dim]

        randomness = np.round(
            self.rng.normal(0, 3.0, size=self.randomness_dim)
        ).astype(np.int64)
        x = np.concatenate([message.astype(np.int64), randomness])
        commitment = self.A @ x % COMMIT_Q
        return commitment, randomness


def _serialize_associated_data(associated_data: Optional[Any]) -> bytes:
    """Canonical bytes for Fiat-Shamir binding (ciphertext / commitment blob)."""
    if associated_data is None:
        return b""
    if isinstance(associated_data, (bytes, bytearray)):
        return bytes(associated_data)
    if isinstance(associated_data, np.ndarray):
        return associated_data.tobytes()
    if isinstance(associated_data, (list, tuple)):
        # list of HE ciphertext dicts {'c0','c1'}
        parts = []
        for item in associated_data:
            if isinstance(item, dict) and "c0" in item and "c1" in item:
                parts.append(np.asarray(item["c0"]).tobytes())
                parts.append(np.asarray(item["c1"]).tobytes())
            elif isinstance(item, np.ndarray):
                parts.append(item.tobytes())
            else:
                parts.append(repr(item).encode())
        return b"".join(parts)
    if isinstance(associated_data, dict) and "c0" in associated_data:
        return (
            np.asarray(associated_data["c0"]).tobytes()
            + np.asarray(associated_data["c1"]).tobytes()
        )
    return repr(associated_data).encode()


class ZKPNormBound:
    """ZKP that ||Δw||₂ ≤ τ, optionally bound to associated_data (e.g. HE cts)."""

    QUANT_SCALE = 1000.0

    def __init__(self, dim: int, threshold: float, seed: int = 42):
        self.dim = dim
        self.tau = threshold
        self.rng = np.random.default_rng(seed)
        self.commitment_scheme = LatticeCommitment(dim, seed)
        # Rejection sampling is over *quantized* integers (gradient * QUANT_SCALE)
        self.sigma_mask = self.tau * self.QUANT_SCALE * REJECTION_BOUND
        self.B_reject = self.sigma_mask * np.sqrt(dim) * 1.5

    def _fiat_shamir_challenge(self, *args) -> int:
        h = hashlib.sha3_256()
        for arg in args:
            if isinstance(arg, np.ndarray):
                h.update(arg.tobytes())
            elif isinstance(arg, (bytes, bytearray)):
                h.update(arg)
            elif isinstance(arg, (int, float)):
                h.update(repr(arg).encode())
            else:
                h.update(repr(arg).encode())
        digest = h.digest()
        return int.from_bytes(digest[:4], "little") % 256 + 1

    def _quantize_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return np.round(gradient * self.QUANT_SCALE).astype(np.int64)

    def generate_proof(
        self,
        gradient: np.ndarray,
        associated_data: Optional[Any] = None,
    ) -> Dict:
        """
        Prove norm bound on `gradient` (length must match self.dim).

        associated_data: BFV ciphertext(s) or bytes mixed into Fiat-Shamir,
        binding the proof to the encrypted payload.
        """
        t_start = time.perf_counter()
        assoc = _serialize_associated_data(associated_data)

        if len(gradient) != self.dim:
            raise ValueError(
                f"ZKP dim mismatch: got {len(gradient)}, expected {self.dim}. "
                "Prove and encrypt the same coordinate slice."
            )

        dw = self._quantize_gradient(gradient)
        actual_norm = float(np.linalg.norm(gradient))

        C, r_commit = self.commitment_scheme.commit(dw)

        y = np.round(self.rng.normal(0, self.sigma_mask, size=self.dim)).astype(np.int64)
        T, r_mask = self.commitment_scheme.commit(y)

        # Bind ciphertext (associated_data) into the challenge
        c = self._fiat_shamir_challenge(C, T, self.tau, assoc)

        z = y + c * dw
        r_z = (r_mask + c * r_commit) % COMMIT_Q
        z_norm = float(np.linalg.norm(z.astype(np.float64)))
        accepted = z_norm <= self.B_reject

        max_attempts = 10
        attempts = 1
        while not accepted and attempts < max_attempts:
            y = np.round(
                self.rng.normal(0, self.sigma_mask, size=self.dim)
            ).astype(np.int64)
            T, r_mask = self.commitment_scheme.commit(y)
            c = self._fiat_shamir_challenge(C, T, self.tau, assoc)
            z = y + c * dw
            r_z = (r_mask + c * r_commit) % COMMIT_Q
            z_norm = float(np.linalg.norm(z.astype(np.float64)))
            accepted = z_norm <= self.B_reject
            attempts += 1

        t_elapsed = time.perf_counter() - t_start

        return {
            "C": C,
            "T": T,
            "z": z,
            "r_z": r_z,
            "z_norm": z_norm,
            "c": c,
            "accepted": accepted,
            "attempts": attempts,
            # Logging only — NEVER use for acceptance decisions
            "actual_norm": actual_norm,
            "generation_time": t_elapsed,
            "proof_size_bytes": int(
                C.nbytes + T.nbytes + z.nbytes + r_z.nbytes + 32 + len(assoc)
            ),
            "associated_data_len": len(assoc),
        }

    def verify_proof(
        self,
        proof: Dict,
        associated_data: Optional[Any] = None,
    ) -> Tuple[bool, float]:
        """Cryptographic verification only (no trusted client Booleans)."""
        t_start = time.perf_counter()
        assoc = _serialize_associated_data(associated_data)

        z = proof["z"]
        z_norm = float(np.linalg.norm(z.astype(np.float64)))
        # Verifier recomputes ||z|| ≤ B — do NOT trust proof["accepted"]
        norm_check = z_norm <= self.B_reject

        c_recomputed = self._fiat_shamir_challenge(
            proof["C"], proof["T"], self.tau, assoc
        )
        challenge_check = c_recomputed == proof["c"]

        z_padded = z.copy()
        if len(z_padded) < self.dim:
            z_padded = np.pad(z_padded, (0, self.dim - len(z_padded)))
        elif len(z_padded) > self.dim:
            z_padded = z_padded[: self.dim]

        r_z = proof["r_z"]
        lhs_input = np.concatenate(
            [z_padded.astype(np.int64), r_z.astype(np.int64)]
        )

        if len(lhs_input) == self.commitment_scheme.total_cols:
            lhs = self.commitment_scheme.A @ lhs_input % COMMIT_Q
            rhs = (
                proof["T"].astype(np.int64)
                + proof["c"] * proof["C"].astype(np.int64)
            ) % COMMIT_Q
            algebraic_check = bool(np.array_equal(lhs, rhs))
        else:
            algebraic_check = False

        t_elapsed = time.perf_counter() - t_start
        is_valid = norm_check and challenge_check and algebraic_check
        return is_valid, t_elapsed


class ZKPBatchNormBound:
    """Batched ZKP over chunks (each chunk should still bind its own cts)."""

    def __init__(
        self, total_dim: int, threshold: float, chunk_size: int = 512, seed: int = 42
    ):
        self.total_dim = total_dim
        self.threshold = threshold
        self.chunk_size = min(chunk_size, total_dim)
        self.n_chunks = (total_dim + chunk_size - 1) // chunk_size
        self.chunk_threshold = threshold / np.sqrt(self.n_chunks) * 1.5
        self.provers = [
            ZKPNormBound(
                min(chunk_size, total_dim - i * chunk_size),
                self.chunk_threshold,
                seed + i,
            )
            for i in range(self.n_chunks)
        ]

    def generate_batch_proof(
        self, gradient: np.ndarray, associated_data_chunks: Optional[list] = None
    ) -> Dict:
        t_start = time.perf_counter()
        proofs = []
        total_proof_size = 0

        for i, prover in enumerate(self.provers):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.total_dim)
            chunk = gradient[start_idx:end_idx]
            assoc = None
            if associated_data_chunks is not None:
                assoc = associated_data_chunks[i]
            proof = prover.generate_proof(chunk, associated_data=assoc)
            proofs.append(proof)
            total_proof_size += proof["proof_size_bytes"]

        t_elapsed = time.perf_counter() - t_start
        return {
            "chunk_proofs": proofs,
            "n_chunks": self.n_chunks,
            "total_generation_time": t_elapsed,
            "total_proof_size_bytes": total_proof_size,
            "all_accepted": all(p["accepted"] for p in proofs),
        }

    def verify_batch_proof(
        self, batch_proof: Dict, associated_data_chunks: Optional[list] = None
    ) -> Tuple[bool, float]:
        t_start = time.perf_counter()
        all_valid = True
        for i, (prover, proof) in enumerate(
            zip(self.provers, batch_proof["chunk_proofs"])
        ):
            assoc = None
            if associated_data_chunks is not None:
                assoc = associated_data_chunks[i]
            valid, _ = prover.verify_proof(proof, associated_data=assoc)
            if not valid:
                all_valid = False
                break
        return all_valid, time.perf_counter() - t_start


def benchmark_zkp(dim=512, threshold=1.0, n_trials=5):
    """Benchmark ZKP with dummy associated_data binding."""
    results = {
        "gen_times": [],
        "ver_times": [],
        "proof_sizes": [],
        "detection_honest": [],
        "detection_malicious": [],
    }

    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        dummy_ct = rng.integers(0, 2**16, size=(64,), dtype=np.int64).tobytes()

        honest_grad = rng.normal(0, threshold / np.sqrt(dim) * 0.5, size=dim)
        zkp = ZKPNormBound(dim, threshold, seed=trial)
        proof = zkp.generate_proof(honest_grad, associated_data=dummy_ct)
        results["gen_times"].append(proof["generation_time"])

        valid, ver_time = zkp.verify_proof(proof, associated_data=dummy_ct)
        results["ver_times"].append(ver_time)
        results["proof_sizes"].append(proof["proof_size_bytes"])
        results["detection_honest"].append(valid)

        # Wrong associated_data must fail binding
        bad_bind, _ = zkp.verify_proof(proof, associated_data=b"tampered")
        assert not bad_bind

        malicious_grad = rng.normal(0, threshold * 10 / np.sqrt(dim), size=dim)
        malicious_grad *= 10
        proof_mal = zkp.generate_proof(malicious_grad, associated_data=dummy_ct)
        valid_mal, _ = zkp.verify_proof(proof_mal, associated_data=dummy_ct)
        # Large updates should fail rejection/norm checks with high probability
        results["detection_malicious"].append(not valid_mal)

    return {
        "gen_time_mean": float(np.mean(results["gen_times"])),
        "ver_time_mean": float(np.mean(results["ver_times"])),
        "proof_size_mean": float(np.mean(results["proof_sizes"])),
        "honest_acceptance_rate": float(np.mean(results["detection_honest"])),
        "malicious_detection_rate": float(np.mean(results["detection_malicious"])),
    }
