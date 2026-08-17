"""
Bit-level Keccak-f[1600] + SHA3-256 sponge (FIPS 202), matching hashlib.

Used as the executable counterpart of formal/easycrypt/lib/KeccakF1600.ec.
"""

from __future__ import annotations

from typing import List

# Rotation offsets ρ (FIPS 202 Table 2), indexed by (x,y)
_RHO = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# Round constants (FIPS 202 Table 1)
_RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]

MASK64 = (1 << 64) - 1


def _rotl64(x: int, n: int) -> int:
    n %= 64
    return ((x << n) | (x >> (64 - n))) & MASK64


def _bytes_to_state(b: bytes) -> List[List[int]]:
    assert len(b) == 200
    st = [[0] * 5 for _ in range(5)]
    for y in range(5):
        for x in range(5):
            lane = 0
            off = 8 * (x + 5 * y)
            for z in range(8):
                lane |= b[off + z] << (8 * z)
            st[x][y] = lane
    return st


def _state_to_bytes(st: List[List[int]]) -> bytes:
    out = bytearray(200)
    for y in range(5):
        for x in range(5):
            lane = st[x][y] & MASK64
            off = 8 * (x + 5 * y)
            for z in range(8):
                out[off + z] = (lane >> (8 * z)) & 0xFF
    return bytes(out)


def keccak_f1600(state_bytes: bytes) -> bytes:
    """Bit-level Keccak-f[1600] on a 200-byte state."""
    A = _bytes_to_state(state_bytes)
    for ir in range(24):
        # θ
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rotl64(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]
        # ρ and π combined into B
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rotl64(A[x][y], _RHO[x][y])
        # χ
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
                A[x][y] &= MASK64
        # ι
        A[0][0] ^= _RC[ir]
        A[0][0] &= MASK64
    return _state_to_bytes(A)


def _sha3_256_sponge(msg: bytes) -> bytes:
    """SHA3-256 sponge: rate=136 bytes, domain 0x06, pad 0x80."""
    rate = 136
    state = bytearray(200)
    # absorb
    offset = 0
    while offset + rate <= len(msg):
        block = msg[offset : offset + rate]
        for i, v in enumerate(block):
            state[i] ^= v
        state = bytearray(keccak_f1600(bytes(state)))
        offset += rate
    # final block + padding
    block = bytearray(msg[offset:])
    block.append(0x06)
    while len(block) < rate:
        block.append(0x00)
    block[-1] |= 0x80
    for i, v in enumerate(block):
        state[i] ^= v
    state = bytearray(keccak_f1600(bytes(state)))
    return bytes(state[:32])


def sha3_256(msg: bytes) -> bytes:
    return _sha3_256_sponge(msg)
