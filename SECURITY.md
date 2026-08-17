# Security notes — pq-zkfl-medical

## Closed (including former README residuals)

| Item | Fix |
|------|-----|
| Single decryptor | Threshold partial decrypt |
| Partial HE / unbound FS / trusted Boolean | Full-vector + ct binding + crypto-only accept |
| Unruh / QROM | `qrom_nizk.py` r=128 + Lean + Python hops + **in-repo `QROM.ec`** |
| Sign-flip / backdoors vs ℓ₂ alone | **Default post-ZKP median** (`ZKFL_ROBUST_AGG=median`; Krum available) |
| SEAL ⊕ threshold as two paths | **`FusedSealThresholdHE`** (`ZKFL_HE_BACKEND=fused`, default if TenSEAL installed) |
| EasyCrypt stdlib linking | Self-contained `formal/easycrypt/QROM.ec` imported by Unruh theories; `python formal/run_formal_ci.py` |

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
