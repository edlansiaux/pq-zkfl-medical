# Security notes — pq-zkfl-medical

## Closed (including former README residuals)

| Item | Fix |
|------|-----|
| Single decryptor | Threshold partial decrypt |
| Partial HE / unbound FS / trusted Boolean | Full-vector + ct binding + crypto-only accept |
| Unruh / QROM / SHA3 | `qrom_nizk.py` r=128 + Lean + Python hops + **SHA3-QROM lib** (`formal/easycrypt/lib/{SHA3,QROMCore}.ec`) |
| Sign-flip / backdoors vs ℓ₂ alone | **Default post-ZKP median** (`ZKFL_ROBUST_AGG=median`; Krum available) |
| SEAL ⊕ threshold as two paths | **`FusedSealThresholdHE`** (`ZKFL_HE_BACKEND=fused`, default if TenSEAL installed) |
| EasyCrypt SHA3-QROM library | `lib/SHA3.ec` + `lib/QROMCore.ec` (`H:=sha3_256`); `python formal/run_formal_ci.py` |
| Imaging CNN (no compact head) | **`ConvNet28`** + `experiments/run_medmnist_cnn.py` |

## Commands

```bash
pip install -r requirements.txt
pip install tenseal medmnist   # optional

python formal/run_formal_ci.py
python experiments/smoke_residuals.py
python experiments/run_target_protocol.py   # fused HE + median by default

# Overrides:
# set ZKFL_HE_BACKEND=numpy|fused|tenseal
# set ZKFL_ROBUST_AGG=median|krum|mean
```
