"""
Round transcript binding: H(prev transcript) enters next Unruh / Enc AD.

Prevents silent swapping of accepted (ct, π) across rounds without detection
in the Fiat–Shamir / Unruh digest.
"""

from __future__ import annotations

import hashlib
from typing import Optional


def _sha3(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p)
    return h.digest()


class RoundTranscript:
    def __init__(self, seed_label: bytes = b"ZKFL-TRANSCRIPT"):
        self.state = _sha3(seed_label)
        self.round_id = 0

    def absorb(self, *parts: bytes) -> bytes:
        self.state = _sha3(b"ABSORB", self.state, *parts)
        return self.state

    def advance(self, round_blob: bytes) -> bytes:
        self.state = _sha3(b"ROUND", self.round_id.to_bytes(4, "little"), self.state, round_blob)
        self.round_id += 1
        return self.state

    def binding(self) -> bytes:
        return b"TX:" + self.state
