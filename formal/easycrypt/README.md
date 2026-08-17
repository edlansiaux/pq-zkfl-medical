# EasyCrypt QROM formalization for ZKFL-PQ Unruh NIZK

## Files

| File | Content |
|------|---------|
| `QROM.ec` | **In-repo** QROM interface (stdlib-free); imported by Unruh theories |
| `UnruhBinaryQROM.ec` | QROM soundness theorem statement + game-hop sketch for Unruh binary NIZK |
| `UnruhBinaryCounting.ec` | Combinatorial \(2^{-r}\) lemma |
| `../run_formal_ci.py` | One-shot CI: Python checkers + Lean + static EC QROM link |
| `../check_unruh_qrom_games.py` | Executable game hops G0–G3 (CI-checkable without EasyCrypt) |
| `../check_unruh_soundness.py` | Exhaustive counting checker |

## How to machine-check

```bash
# Recommended (no EasyCrypt binary required):
python formal/run_formal_ci.py

# Or individually:
python -m formal.check_unruh_soundness
python -m formal.check_unruh_qrom_games

# With EasyCrypt installed (opam / easycrypt):
cd formal/easycrypt
easycrypt -I . UnruhBinaryCounting.ec
easycrypt -I . UnruhBinaryQROM.ec
```

Lean + Python hops are always CI-checkable. EasyCrypt stdlib QROM linking is optional for users who have the binary; theories already `require import QROM` from this directory.

## Mapping to code

`crypto/qrom_nizk.py` implements the Unruh transform with default `r=128`, invertible RO records `(ρ, H(ρ))`, and ciphertext-bound challenges — the concrete instantiation of the theorems above.
