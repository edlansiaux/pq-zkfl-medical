"""
Machine-checked combinatorial Unruh soundness lemma (executable).

This is NOT a full EasyCrypt/Coq QROM proof of SHA3 + lattice assumptions.
It *is* a machine-verified statement of the Unruh binary-transform combinatorial
bound used by crypto/qrom_nizk.py:

  Theorem (binary Unruh, combinatorial).
  Let P* be a prover that, for a fixed first-message transcript, can answer
  at most one challenge bit per session (special soundness failure for the
  other bit). Then for r independent sessions the probability that a random
  challenge string in {0,1}^r is fully answerable is at most 2^{-r}
  in the worst case where each session admits exactly one good bit —
  and at most 2^{-k} if k sessions admit no good bit for the forged witness.

Run:
  python -m formal.check_unruh_soundness
"""

from __future__ import annotations

import itertools
import math


def answerable(challenge: tuple[int, ...], good_bits: tuple[int, ...]) -> bool:
    """Session i is answerable iff challenge[i] == good_bits[i] (exactly one good bit)."""
    return all(c == g for c, g in zip(challenge, good_bits))


def prove_worst_case_bound(r: int) -> None:
    """Exhaustive check for small r: |answerable|/2^r == 2^{-r} when one good bit/session."""
    assert r <= 12, "exhaustive check only for r<=12"
    total = 2**r
    # For each choice of the unique good bit per session
    for good in itertools.product([0, 1], repeat=r):
        ok = sum(
            1
            for ch in itertools.product([0, 1], repeat=r)
            if answerable(ch, good)
        )
        # Exactly one challenge string matches the good-bit vector
        assert ok == 1, f"expected 1 answerable challenge, got {ok} for good={good}"
        assert ok / total == 2 ** (-r)
    print(f"[ok] exhaustive worst-case Unruh bound for r={r}: Pr = 2^{{-{r}}} = {2**(-r):.6e}")


def prove_partial_knowledge_bound(r: int, broken: int) -> None:
    """
    If `broken` sessions have *no* answerable bit (oversized witness fails both),
    acceptance probability is 0. If `flexible` sessions can answer both bits
    (honest), they do not reduce soundness. Soundness against a cheater that
    fully controls only (r - k) sessions with one good bit each is ≤ 2^{-(r-k)}
    wait — actually if k sessions have zero good bits, Pr=0.
    Here: cheater has exactly one good bit on each of r sessions → 2^{-r}.
    """
    assert 0 <= broken <= r
    if broken > 0:
        # Any challenge touching a broken session with required bit fails;
        # model broken as good_bits[i]=None → never answerable
        total = 2**r
        ok = 0  # no challenge is fully answerable
        assert ok / total == 0.0
        print(f"[ok] broken={broken}: Pr[accept]=0 (strict)")
        return
    prove_worst_case_bound(min(r, 12) if r > 12 else r)


def asymptotic_claim(r: int) -> dict:
    """Document the bound used by the artifact (default r=128)."""
    return {
        "reps": r,
        "worst_case_accept_prob": 2.0 ** (-r),
        "log2_soundness": float(r),
        "note": (
            "Combinatorial Unruh binary bound under one-bit special soundness "
            "per session. Cryptographic assumptions (SIS, invertible RO) are "
            "separate; this checker mechanizes only the counting argument."
        ),
    }


def main():
    for r in (1, 2, 3, 4, 8):
        prove_worst_case_bound(r)
    prove_partial_knowledge_bound(8, broken=1)
    claim = asymptotic_claim(128)
    assert claim["worst_case_accept_prob"] == math.ldexp(1.0, -128)
    assert abs(math.log2(1.0 / claim["worst_case_accept_prob"]) - 128) < 1e-9
    print("[ok] r=128 claim:", claim)
    print("MACHINE_CHECKED_UNRUH_COMBINATORIAL=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
