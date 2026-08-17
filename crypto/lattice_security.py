"""
Rough lattice-security reporting for the BFV parameter presets.

Uses the HomomorphicEncryption.org / Albrecht et al. style *guidance*:
security grows with ring degree n and modulus bit-length. For a certified
estimate, plug (n, q, secret distribution) into the lattice estimator
(https://github.com/malb/lattice-estimator). This module documents the
preset and, when `lattice_estimator` is installed, prints a real estimate.
"""

from __future__ import annotations

from crypto.homomorphic import (
    HE_CLAIMED_SECURITY_BITS,
    HE_N,
    HE_PRESET,
    HE_Q,
    HE_T,
    _HE_PRESETS,
)


def report_he_security() -> dict:
    out = {
        "preset": HE_PRESET,
        "n": HE_N,
        "log2_q": int(HE_Q).bit_length(),
        "t": HE_T,
        "claimed_bits": HE_CLAIMED_SECURITY_BITS,
        "estimator": None,
        "note": (
            "HomomorphicEncryption.org Classic-128 lists n=4096-class rings. "
            "Set ZKFL_HE_PRESET=classic128 for that ring degree."
        ),
    }

    try:
        # Optional dependency — not required
        from estimator import LWE, ND, partial  # type: ignore

        # Very rough: treat as RLWE ≈ LWE with dimension n
        params = LWE.Parameters(
            n=HE_N,
            q=HE_Q,
            Xs=ND.UniformMod(3),
            Xe=ND.DiscreteGaussian(3.2),
        )
        # lattice-estimator API varies by version; best-effort
        try:
            from estimator.lwe_estimate import estimate  # type: ignore

            est = estimate(params)
            out["estimator"] = str(est)
        except Exception as e:  # noqa: BLE001
            out["estimator"] = f"available but estimate failed: {e}"
    except ImportError:
        out["estimator"] = "lattice_estimator not installed (pip optional)"

    out["presets"] = {k: {"n": v["n"], "claimed_bits": v["claimed_bits"]} for k, v in _HE_PRESETS.items()}
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(report_he_security(), indent=2))
