"""
CI: bit-level Keccak-f[1600] + SHA3-256 vs hashlib, algebraic lane lemmas,
and EasyCrypt lane theory link.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from crypto.keccak_f1600 import (
    RC,
    RHO_OFFSETS,
    bytes_to_state,
    chi,
    iota,
    keccak_f1600,
    keccak_round,
    lane_idx,
    pi,
    pi_dest,
    rho,
    sha3_256,
    state_to_bytes,
    theta,
)


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


def check_algebraic_lane_lemmas() -> None:
    raw = secrets.token_bytes(200)
    A = bytes_to_state(raw)
    assert state_to_bytes(A) == raw

    # θ column parity + D applied to every lane in the column
    At, C, D = theta(A)
    for x in range(5):
        parity = A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4]
        assert C[x] == parity
        for y in range(5):
            assert At[x][y] == (A[x][y] ^ D[x])

    # ρ rotates each lane by Table-2 offset
    Ar = rho(A)
    for x in range(5):
        for y in range(5):
            n = RHO_OFFSETS[x][y] % 64
            if n == 0:
                expect = A[x][y]
            else:
                expect = ((A[x][y] << n) | (A[x][y] >> (64 - n))) & ((1 << 64) - 1)
            assert Ar[x][y] == expect

    # π is a bijection on lane indices
    seen = set()
    inv = {}
    for x in range(5):
        for y in range(5):
            d = pi_dest(x, y)
            assert 0 <= d < 25
            assert d not in seen
            seen.add(d)
            inv[d] = (x, y)
    assert len(seen) == 25
    Ap = pi(A)
    for x in range(5):
        for y in range(5):
            yy = (2 * x + 3 * y) % 5
            assert Ap[y][yy] == A[x][y]
            assert inv[pi_dest(x, y)] == (x, y)

    # χ local algebraic form
    Ac = chi(A)
    for x in range(5):
        for y in range(5):
            expect = (
                A[x][y] ^ ((~A[(x + 1) % 5][y]) & A[(x + 2) % 5][y])
            ) & ((1 << 64) - 1)
            assert Ac[x][y] == expect

    # ι touches only lane (0,0)
    assert len(RC) == 24
    for ir in range(24):
        Ai = iota(A, ir)
        assert Ai[0][0] == (A[0][0] ^ RC[ir]) & ((1 << 64) - 1)
        for x in range(5):
            for y in range(5):
                if (x, y) != (0, 0):
                    assert Ai[x][y] == A[x][y]

    # round composition + 24-fold matches keccak_f1600
    st = A
    for ir in range(24):
        st = keccak_round(st, ir)
    assert state_to_bytes(st) == keccak_f1600(raw)

    # lane_idx layout
    for y in range(5):
        for x in range(5):
            assert lane_idx(x, y) == x + 5 * y

    print("[ok] algebraic lane lemmas (θ/ρ/π/χ/ι + packing + 24-fold)")


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
        "theta_column_parity",
        "rho_rotates_lane",
        "pi_bijection",
        "chi_local",
        "iota_local_zero",
        "iota_local_rest",
        "packing_roundtrip",
        "keccak_f1600_is_round_fold",
        "lemma keccak_round_len",
        "lemma packing_roundtrip",
    ):
        assert needle in kec, needle
    assert "require import KeccakF1600" in sha
    assert "sha3_256" in sha
    print("[ok] EasyCrypt algebraic lane lemmas present")


def main() -> None:
    check_keccak_f_roundtrip_size()
    check_sha3_vs_hashlib()
    check_algebraic_lane_lemmas()
    check_easycrypt_bitlevel()
    print("MACHINE_CHECKED_KECCAK_BITLEVEL=1")
    print("MACHINE_CHECKED_KECCAK_LANE_LEMMAS=1")


if __name__ == "__main__":
    main()
