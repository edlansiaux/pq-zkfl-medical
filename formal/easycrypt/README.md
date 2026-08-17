# EasyCrypt QROM + SHA3 formalization for ZKFL-PQ Unruh NIZK

## Layout

| Path | Content |
|------|---------|
| `lib/KeccakF1600.ec` | Bit-level Keccak-f[1600] + algebraic lane lemmas |
| `lib/SHA3.ec` | FIPS 202 SHA3-256 sponge on Keccak-f[1600] |
| `lib/QROMCore.ec` | Production QROM core: `H := sha3_256`, `qrom_term(q)=q(q+1)/2^{256}` |
| `SHA3_QROM.ec` | Explicit SHA3↔RO link lemma |
| `QROM.ec` | Re-export for Unruh theories |
| `UnruhBinaryQROM.ec` | Unruh soundness shape + SHA3 O2H lemma |
| `UnruhBinaryCounting.ec` | Combinatorial \(2^{-r}\) |
| `../check_sha3_qrom.py` | FIPS empty-message vector + EC link CI |
| `../run_formal_ci.py` | One-shot formal CI |

## Check

```bash
python formal/run_formal_ci.py

# With EasyCrypt installed:
cd formal/easycrypt
easycrypt -I . -I lib UnruhBinaryQROM.ec
```
