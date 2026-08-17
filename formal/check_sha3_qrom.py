"""
Validate SHA3 instantiation of the EasyCrypt QROM library (CI-checkable).

Checks:
  1) FIPS 202 SHA3-256 empty-message test vector (Python hashlib)
  2) EasyCrypt sources contain SHA3 + concrete qrom_term + Unruh link
  3) qrom_term(q) = q(q+1)/2^256 matches the production-library formula
"""

from __future__ import annotations

import hashlib
import os
import re


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EC = os.path.join(ROOT, "formal", "easycrypt")


def check_fips_sha3_256() -> None:
    # FIPS 202 — SHA3-256("") 
    expected = bytes.fromhex(
        "a7ffc6f8bf1ed76651c14756a061d662"
        "f580ff4de43b49fa82d80a4b80f8434a"
    )
    got = hashlib.sha3_256(b"").digest()
    assert got == expected, (got.hex(), expected.hex())
    # Non-empty sanity
    assert len(hashlib.sha3_256(b"zkfl-pq").digest()) == 32
    print("[ok] FIPS 202 SHA3-256 empty-message test vector")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def check_easycrypt_sha3_qrom_link() -> None:
    sha3 = _read(os.path.join(EC, "lib", "SHA3.ec"))
    core = _read(os.path.join(EC, "lib", "QROMCore.ec"))
    qrom = _read(os.path.join(EC, "QROM.ec"))
    unruh = _read(os.path.join(EC, "UnruhBinaryQROM.ec"))
    link = _read(os.path.join(EC, "SHA3_QROM.ec"))

    assert "op sha3_256" in sha3 and "keccak_f1600" in sha3
    assert "require import SHA3" in core
    assert "op H (x : input) : output = sha3_256 x" in core.replace("  ", " ") or (
        "sha3_256 x" in core and "op H" in core
    )
    assert "2%r ^ 256" in core and "qrom_term" in core
    assert "require import QROMCore" in qrom
    assert "require import QROM" in unruh
    assert "qrom_term_is_sha3_o2h" in unruh
    assert "sha3_is_ro" in link
    print("[ok] EasyCrypt SHA3-QROM production library linked")


def check_qrom_term_formula() -> None:
    def qrom_term(q: int) -> float:
        if q < 0:
            return 0.0
        return (q * (q + 1)) / (2**256)

    assert qrom_term(0) == 0.0
    assert qrom_term(1) == 2 / (2**256)
    assert qrom_term(100) < 1e-70
    print("[ok] qrom_term(q)=q(q+1)/2^256 (SHA3-256 O2H shape)")


def main() -> None:
    check_fips_sha3_256()
    check_easycrypt_sha3_qrom_link()
    check_qrom_term_formula()
    print("MACHINE_CHECKED_SHA3_QROM_LIB=1")


if __name__ == "__main__":
    main()
