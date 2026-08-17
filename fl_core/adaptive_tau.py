"""
Adaptive public ℓ₂ threshold τ from accepted-update history.

τ_{t+1} = clip( c · Q_{1-α}({‖Δ‖ : accepted at t}), τ_min, τ_max )
Published and bound into the next round's proof associated data.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class AdaptiveTau:
    def __init__(
        self,
        tau0: float,
        tau_min: float = 1.0,
        tau_max: float = 50.0,
        quantile: float = 0.9,
        scale: float = 1.25,
        ema: float = 0.5,
    ):
        self.tau = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.quantile = float(quantile)
        self.scale = float(scale)
        self.ema = float(ema)
        self.history: List[float] = [self.tau]

    def observe(self, accepted_norms: Sequence[float]) -> float:
        """Update τ from this round's accepted norms; return new public τ."""
        if accepted_norms:
            q = float(np.quantile(np.asarray(accepted_norms, dtype=np.float64), self.quantile))
            proposed = self.scale * q
            new_tau = (1.0 - self.ema) * self.tau + self.ema * proposed
            self.tau = float(np.clip(new_tau, self.tau_min, self.tau_max))
        self.history.append(self.tau)
        return self.tau

    def public_bytes(self) -> bytes:
        return f"TAU:{self.tau:.8f}".encode()
