# Security notes — pq-zkfl-medical

## Closed residuals

| Item | Implementation |
|------|----------------|
| Single decryptor | `(t,n)` ThresholdBFV **partial** decrypt; monolithic `sk` erased after sharegen |
| Partial HE / unbound FS / trusted Boolean | Full-vector encrypt; ct in challenges; crypto-only accept |
| Unruh / QROM / SHA3 | `qrom_nizk.py` (default `r=128`) + Lean + Python hops + EasyCrypt `lib/{SHA3,QROMCore,KeccakF1600}.ec` |
| Shared SIS at large \(d\) | One `LatticeCommitment` for all Unruh sessions (+ base Σ) |
| Sign-flip / in-bound backdoors vs \(\ell_2\) alone | Default post-ZKP **median** (`ZKFL_ROBUST_AGG=median`; Krum available) |
| SEAL ⊕ threshold | `FusedSealThresholdHE` (`ZKFL_HE_BACKEND=fused` if TenSEAL installed) |
| Unruh Enc-consistency | `crypto/unruh_enc_consistency.py` — same Unruh transform as norm proof |
| PartialDecrypt NIZK | `crypto/partial_dec_nizk.py` — ROM-FS; abort on bad \(\mu_i\) (not Unruh-lifted yet) |
| Dual-norm / adaptive \(\tau\) / transcript | `fl_core/clip.py`, `adaptive_tau.py`, `crypto/round_transcript.py` |
| Imaging CNN (no compact head) | `ConvNet28` + `experiments/run_medmnist_cnn.py` |

## Commands

```bash
pip install -r requirements.txt
pip install tenseal medmnist   # optional

python formal/run_formal_ci.py
python experiments/smoke_residuals.py
python experiments/run_innovation_pack.py
python experiments/run_target_protocol.py

# Overrides:
#   ZKFL_HE_BACKEND=numpy|fused|tenseal
#   ZKFL_HE_PRESET=classic128_demo|classic128
#   ZKFL_ROBUST_AGG=median|krum|mean
```

## Out of scope (explicit)

- Differential privacy (compose externally if needed)
- Claiming NumPy `n=512` demo as SEAL-certified Classic-128
- Fully machine-checked EasyCrypt discharge of every Keccak algebraic identity without the FIPS executable checker
- Quoting innovation-pack `r=16` / `r_enc=4` as the library 128-bit Unruh class
- Claiming median eliminates backdoors (residual ASR ≈55% in `results/`)
