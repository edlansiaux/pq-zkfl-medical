# Security notes — pq-zkfl-medical (residuals closed)

## Closed

| Former limitation | Fix |
|-------------------|-----|
| Single decryptor / reconstruct-`sk` | **Partial decryption** (`ThresholdBFV.partial_decrypt`) |
| Partial HE | Full-vector chunking |
| Trusted Boolean / unbound FS | Crypto-only verify + ct binding |
| Classical FS only | **Unruh NIZK** default `r=128` |
| Synthetic-only | UCI Breast Cancer + MedMNIST path |
| No 128-bit HE story | Presets + `lattice_security.py` |
| Enc-consistency of ρ | **Dedicated gadget** `crypto/enc_consistency.py` (Σ-protocol on BFV coins) |
| Homemade-only HE | **Microsoft SEAL** via TenSEAL (`ZKFL_HE_BACKEND=tenseal`) |
| Unruh not machine-checked | **Combinatorial lemma** executable in `formal/check_unruh_soundness.py` |

## Commands

```bash
pip install -r requirements.txt
pip install tenseal medmnist   # optional SEAL backend + imaging

python experiments/smoke_residuals.py
python -m formal.check_unruh_soundness
python experiments/run_target_protocol.py

# SEAL path:
# set ZKFL_HE_BACKEND=tenseal
# python experiments/run_target_protocol.py
```

## Honest scope of the Unruh checker

`formal/check_unruh_soundness.py` machine-checks the **binary Unruh counting bound** (worst-case accept ≤ 2^{-r}). It does **not** replace a full EasyCrypt/Coq QROM proof of SHA3 + SIS for the concrete hash instantiation — that remains research-grade proof engineering outside the workshop artifact.
