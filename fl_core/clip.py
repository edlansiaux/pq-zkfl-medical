"""
Coordinate ℓ∞ clipping before prove+encrypt (dual-norm gate).

Honest clients and the language statement use
  w̃ = Clip_∞(Δ, τ_∞)    then    Prove_ℓ₂(w̃) ∧ Enc(w̃).
Malicious unclipped plaintext fails Enc-consistency vs the proven vector when
wired to the same associated data.
"""

from __future__ import annotations

import numpy as np


def clip_infty(vec: np.ndarray, tau_inf: float) -> np.ndarray:
    t = float(tau_inf)
    if t <= 0:
        raise ValueError("tau_inf must be positive")
    v = np.asarray(vec, dtype=np.float64)
    return np.clip(v, -t, t)


def dual_norm_ok(vec: np.ndarray, tau2: float, tau_inf: float) -> bool:
    v = np.asarray(vec, dtype=np.float64)
    return float(np.linalg.norm(v)) <= float(tau2) + 1e-9 and float(np.max(np.abs(v))) <= float(
        tau_inf
    ) + 1e-9
