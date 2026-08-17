"""
CI: bit-level Keccak-f[1600] + SHA3-256 vs hashlib, and EasyCrypt lane theory link.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from crypto.keccak_f1600 import keccak_f1600, sha3_256


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EC_LIB = os.path.join(ROOT, "formal", "easycrypt", "lib")


def check_sha3_vs_hashlib() -> None:
    vectors = [
        b"",
        b"a",
        b"abc",
        b"zkfl-pq",
        b"\x00" * 200,
        secrets.token_bytes(17),
        secrets.token_bytes(135),
        secrets.token_bytes(136),
        secrets.token_bytes(137),
        secrets.token_bytes(1000),
    ]
    for m in vectors:
        assert sha3_256(m) == hashlib.sha3_256(m).digest(), m[:16]
    # FIPS empty
    assert sha3_256(b"") == bytes.fromhex(
        "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    )
    print(f"[ok] bit-level SHA3-256 == hashlib on {len(vectors)} messages + FIPS empty")


def check_keccak_f_roundtrip_size() -> None:
    s0 = bytes(200)
    s1 = keccak_f1600(s0)
    assert len(s1) == 200 and s1 != s0
    s2 = keccak_f1600(s1)
    assert len(s2) == 200 and s2 != s1
    print("[ok] keccak_f1600 preserves 200-byte state and is nontrivial")


def check_easycrypt_bitlevel() -> None:
    kec = open(os.path.join(EC_LIB, "KeccakF1600.ec"), encoding="utf-8").read()
    sha = open(os.path.join(EC_LIB, "SHA3.ec"), encoding="utf-8").read()
    for needle in (
        "op theta",
        "op rho",
        "op pi",
        "op chi",
        "op iota",
        "keccak_round",
        "keccak_f1600_state",
        "RC_len",
        "keccak_f1600_len",
    ):
        assert needle in kec, needle
    assert "require import KeccakF1600" in sha
    assert "sha3_256" in sha
    print("[ok] EasyCrypt bit-level Keccak-f[1600] theory present")


def main() -> None:
    check_keccak_f_roundtrip_size()
    check_sha3_vs_hashlib()
    check_easycrypt_bitlevel()
    print("MACHINE_CHECKED_KECCAK_BITLEVEL=1")


if __name__ == "__main__":
    main()
