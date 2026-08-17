# Security notes — pq-zkfl-medical (all prior residuals closed)

## Closed

| Former limitation | Fix |
|-------------------|-----|
| Single decryptor / reconstruct-`sk` | **Partial decryption** (`ThresholdBFV.partial_decrypt`) |
| Partial HE | Full-vector chunking |
| Trusted Boolean / unbound FS | Crypto-only verify + ct binding |
| Classical FS only | **Unruh NIZK** default `r=128` |
| Synthetic-only / scale | UCI + **MedMNIST full-res** + **N=20 T=30** (`run_scale.py`) |
| Backdoors (former non-goal) | Measured + **`hybrid_zkp_median`** (`run_backdoor.py`) |
| No 128-bit HE story | Presets + `lattice_security.py` + TenSEAL |
| Enc-consistency of ρ | `crypto/enc_consistency.py` |
| Homemade-only HE | `ZKFL_HE_BACKEND=tenseal` |
| Unruh / QROM not machine-checked | **Lean 4** (`formal/lean`, `lake build`) + Python game hops + EasyCrypt sources |

## Commands

```bash
pip install -r requirements.txt
pip install tenseal medmnist

python experiments/smoke_residuals.py
python -m formal.check_unruh_soundness
python -m formal.check_unruh_qrom_games
cd formal/lean && lake build

python experiments/run_scale.py
python experiments/run_backdoor.py
python experiments/run_medmnist_fullres.py
python experiments/run_target_protocol.py
```

## Results pointers

- `results/scale_results.json` — N=20, T=30
- `results/backdoor_results.json` — ASR drop with hybrid_zkp_median
- `results/medmnist_fullres_results.json` — 784-D, no projection
