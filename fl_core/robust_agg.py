"""
Robust aggregation helpers composed with ZKFL-PQ after cryptographic accept.

Default post-ZKP aggregator is coordinate-wise median (resilient to sign-flip /
in-bound backdoors that pass ell_2 gates). Multi-Krum is available via env.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


def coord_median(deltas: List[np.ndarray]) -> np.ndarray:
    if not deltas:
        raise ValueError("empty delta list")
    return np.median(np.stack(deltas, axis=0), axis=0)


def multi_krum(deltas: List[np.ndarray], f: int = 1) -> np.ndarray:
    n = len(deltas)
    if n == 0:
        raise ValueError("empty delta list")
    if n <= 2 * f + 2:
        return np.mean(deltas, axis=0)
    scores = []
    for i, di in enumerate(deltas):
        dists = sorted(
            float(np.linalg.norm(di - dj)) for j, dj in enumerate(deltas) if i != j
        )
        scores.append((sum(dists[: n - f - 2]), i))
    scores.sort()
    keep = [deltas[i] for _, i in scores[: n - f]]
    return np.mean(keep, axis=0)


def robust_aggregate(
    deltas: List[np.ndarray],
    method: Optional[str] = None,
    f: int = 1,
) -> np.ndarray:
    """
    Post-ZKP aggregator.

    method:
      - median (default): coordinate-wise median
      - krum: Multi-Krum then mean of selected
      - mean: plain mean (legacy / ablation)
    Override with ZKFL_ROBUST_AGG=median|krum|mean
    """
    if method is None:
        method = os.environ.get("ZKFL_ROBUST_AGG", "median").lower()
    if not deltas:
        raise ValueError("empty delta list")
    if method in ("median", "hybrid_zkp_median"):
        return coord_median(deltas)
    if method in ("krum", "multi_krum", "hybrid_zkp_krum"):
        return multi_krum(deltas, f=f)
    if method in ("mean", "avg", "fedavg"):
        return np.mean(deltas, axis=0)
    raise ValueError(f"unknown robust agg method: {method}")
