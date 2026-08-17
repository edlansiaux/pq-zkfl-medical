# EasyCrypt QROM formalization for ZKFL-PQ Unruh NIZK

## Files

| File | Content |
|------|---------|
| `UnruhBinaryQROM.ec` | QROM soundness theorem statement + game-hop sketch for Unruh binary NIZK |
| `UnruhBinaryCounting.ec` | Combinatorial \(2^{-r}\) lemma |
| `../check_unruh_qrom_games.py` | Executable game hops G0–G3 (CI-checkable without EasyCrypt) |
| `../check_unruh_soundness.py` | Exhaustive counting checker |

## How to machine-check

```bash
# Always available (Python):
python -m formal.check_unruh_soundness
python -m formal.check_unruh_qrom_games

# With EasyCrypt installed (opam / easycrypt):
cd formal/easycrypt
easycrypt UnruhBinaryCounting.ec
easycrypt UnruhBinaryQROM.ec
```

## Mapping to code

`crypto/qrom_nizk.py` implements the Unruh transform with default `r=128`, invertible RO records `(ρ, H(ρ))`, and ciphertext-bound challenges — the concrete instantiation of the theorems above.
