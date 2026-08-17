# Security notes — pq-zkfl-medical (limitations closed)

## Closed

| Former limitation | Fix |
|-------------------|-----|
| Single decryptor / reconstruct-`sk` | **Partial decryption**: each party sends `μ_i = c1 ⋆ (λ_i·s_i)`; combine without assembling `s` (`ThresholdBFV.partial_decrypt`) |
| Partial HE | Full-vector chunking |
| Trusted Boolean / unbound FS | Crypto-only verify + ct binding |
| Classical FS only | **Unruh NIZK** default `r=128` (`qrom_nizk.py`) |
| Synthetic-only | UCI Breast Cancer + **MedMNIST** (`run_medmnist.py`) |
| No 128-bit HE story | Presets `classic128` (`n=4096`) / `classic128_demo` (`n=512`) + `lattice_security.py` |

## Commands

```bash
pip install -r requirements.txt
pip install medmnist   # optional imaging

python -m crypto.lattice_security
python experiments/run_target_protocol.py
python experiments/run_medmnist.py

# Full Classic-128 ring (slow on NumPy):
# set ZKFL_HE_PRESET=classic128
```

## Remaining (optional certification)

- Plug parameters into [lattice-estimator](https://github.com/malb/lattice-estimator) / SEAL for an external certificate (reporter supports the package when installed).
- Formal Coq/EasyCrypt proof of the Unruh instantiation (the transform *is* implemented; a machine-checked proof is a separate artifact).
