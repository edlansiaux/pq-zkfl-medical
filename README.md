# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Code and experiments for:

> **Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI**  
> Edouard Lansiaux

**ZKFL-PQ** is a layered protocol for medical federated learning:

1. **ML-KEM-768** (FIPS 203) — post-quantum transport
2. **Unruh NIZK** of $\ell_2$-bounded updates in $\mathcal{L}_{\tau}^{\mathrm{bind}}$ + **Enc-consistency** (coins $\rho$) — ciphertext-bound proofs
3. **BFV full-vector HE** with **$(t,n)$ threshold partial decrypt** — no monolithic server `sk`
4. **Default post-ZKP median** (`ZKFL_ROBUST_AGG=median`; Multi-Krum available) — against sign-flip and in-bound backdoors that pure $\ell_2$ proofs accept

Related secure-aggregation stacks such as Beskar are complementary; see [`SECURITY.md`](SECURITY.md).

### Key Results (target protocol / UCI Breast Cancer)

| Metric | Value |
|--------|-------|
| Final accuracy (3 rounds) | **93.9%** |
| Large-norm detection (attack round) | **yes** |
| Defaults | **fused** HE + **median** aggregation |
| Payload / round | ≈ 1.4–1.7 MB (full-vector HE) |
| Unruh default | **r=128** (latency-sensitive runs often 32–64) |
| Threshold | **(2,3) partial decrypt** (`sk` never assembled) |
| Backdoor ASR | FedAvg/$\ell_2$ ≈98% → **ZKP+median ≈55%** (clean ≈99%) |

```bash
python experiments/run_target_protocol.py   # fused HE + median by default
python experiments/run_baselines.py
python experiments/run_backdoor.py
python experiments/run_medmnist_fullres.py  # needs medmnist
python formal/run_formal_ci.py              # Lean + Python + QROM.ec link
```

## Security Notes

- **ZKP verification**: algebraic check `A·[z || r_z] ≡ T + c·C mod q`, plus Fiat–Shamir / Unruh binding to BFV ciphertext bytes (`associated_data`).
- **No trusted Boolean**: acceptance does **not** use client-supplied `is_within_bound`.
- **Full-vector HE**: all coordinates encrypted (chunked); ZKP covers the same vector.
- **Threshold BFV**: server `sk=None` after share generation; open via Lagrange-weighted partial decryptions.
- **Enc-consistency**: `crypto/enc_consistency.py` proves knowledge of BFV coins ρ for each ciphertext.
- **HE backend** (default if TenSEAL installed): `ZKFL_HE_BACKEND=fused` → `FusedSealThresholdHE` (NumPy threshold + SEAL sidecar). Overrides: `numpy` | `tenseal`.
- **Robust aggregation** (default): `ZKFL_ROBUST_AGG=median` (also `krum` | `mean`).
- **Unruh NIZK**: default `r=128`; Lean uniqueness + Python game hops + in-repo `formal/easycrypt/QROM.ec` via `python formal/run_formal_ci.py`.
- **HE presets**: `ZKFL_HE_PRESET=classic128_demo` (n=512) or `classic128` (n=4096); see `crypto/lattice_security.py`.

## Repository Structure

```
pq-zkfl-medical/
├── crypto/
│   ├── ml_kem.py              # ML-KEM-768
│   ├── zkp_norm.py            # Σ-protocol + algebraic verify
│   ├── qrom_nizk.py           # Unruh NIZK (r=128 default)
│   ├── enc_consistency.py     # BFV coins Σ-gadget
│   ├── homomorphic.py         # BFV + ThresholdBFV partial decrypt
│   ├── fused_he.py            # Fused SEAL + threshold (default path)
│   ├── seal_backend.py        # Optional TenSEAL/SEAL
│   └── lattice_security.py    # HE presets / estimator report
├── fl_core/
│   ├── model.py               # MLP + synthetic / UCI / MedMNIST
│   ├── cnn.py                 # ConvNet28 (no compact head)
│   └── robust_agg.py          # median / Multi-Krum / mean
├── experiments/
│   ├── run_target_protocol.py # Full stack (defaults: fused + median)
│   ├── run_baselines.py       # Multi-seed FedAvg / clip / Krum / hybrid
│   ├── run_backdoor.py        # Trigger ASR (+ hybrid_zkp_median)
│   ├── run_scale.py           # N=20, T=30
│   ├── run_medmnist.py / run_medmnist_fullres.py
│   ├── run_medmnist_cnn.py    # ConvNet28 + HE + Unruh
│   ├── make_excellence_figure.py  # Backdoor ASR figure helper
│   └── smoke_residuals.py
├── formal/
│   ├── run_formal_ci.py       # One-shot CI (incl. SHA3-QROM)
│   ├── check_sha3_qrom.py
│   ├── check_unruh_*.py
│   ├── lean/
│   └── easycrypt/
│       ├── lib/SHA3.ec, QROMCore.ec
│       └── UnruhBinaryQROM.ec
├── manuscript/ehpwas2026/     # IEEE conference manuscript (≤6 pages)
│   ├── main.tex / main.pdf
│   └── figures/
├── results/                   # JSON logs
├── SECURITY.md
├── requirements.txt
└── README.md
```

## Quick Start

### Requirements

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, cryptography
- Optional: `tenseal` (fused SEAL path), `medmnist`, Lean/`lake` for formal CI

### Installation

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
pip install -r requirements.txt
pip install tenseal medmnist   # optional
```

### Run Experiments

```bash
python experiments/run_target_protocol.py
python experiments/run_baselines.py
python experiments/run_backdoor.py
python experiments/run_medmnist_cnn.py      # ConvNet28, no compact head
python formal/run_formal_ci.py
```

### Compile Manuscript

```bash
cd manuscript/ehpwas2026
# tectonic main.tex   # or pdflatex ×2
```

## Cryptographic Implementations

### ML-KEM-768 (`crypto/ml_kem.py`)
- FIPS 203–oriented Module-LWE KEM: n=256, k=3, q=3329
- KeyGen / Encaps / Decaps + AES-256-CTR payload wrap

### ZKP + Unruh (`crypto/zkp_norm.py`, `crypto/qrom_nizk.py`)
- Σ-protocol with SIS commitments and algebraic verify
- Unruh parallel binary sessions (default `r=128`) with invertible RO records
- Challenges digest ciphertext bytes; Enc-consistency binds coins ρ

### BFV + Fused threshold (`crypto/homomorphic.py`, `crypto/fused_he.py`)
- Full-vector chunking; `(t,n)` ThresholdBFV partial decryption
- Default `fused`: NumPy threshold privacy + TenSEAL sidecar in one process
- Presets: `classic128_demo` (n=512) / `classic128` (n=4096)

## Ablation Studies

### Varying Malicious Clients (0–3)
| # Malicious | Final Accuracy | Detection Rate | False Positives |
|-------------|----------------|----------------|-----------------|
| 0 | 100.0% | N/A | 0 |
| 1 | 100.0% | 100% | 0 |
| 2 | 100.0% | 100% | 0 |
| 3 | 100.0% | 100% | 0 |

### Varying Threshold τ
| τ | Detection Rate | False Positive Rate |
|---|----------------|---------------------|
| 1.0 | 100% | 13.6% |
| 2.0 | 100% | 13.6% |
| 5.0 | 100% | 0% |
| 10.0 | 100% | 0% |
| 50.0 | 100% | 0% |

## Limitations

Bit-level Keccak-$f$[1600] (θ/ρ/π/χ/ι) is in `formal/easycrypt/lib/KeccakF1600.ec` with FIPS equivalence via `formal/check_keccak_bitlevel.py`. Full-vector HE+Unruh scales with $d$; a shared SIS commitment key and modular BFV mul keep ConvNet28-scale runs interactive. Overrides: `ZKFL_HE_BACKEND`, `ZKFL_ROBUST_AGG`.

## Citation

```bibtex
@article{lansiaux2026zkflpq,
  title={Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI},
  author={Lansiaux, Edouard},
  journal={arXiv preprint arXiv:2603.03398},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Edouard Lansiaux** — [edouard.lansiaux@chu-lille.fr](mailto:edouard.lansiaux@chu-lille.fr)
- STaR-AI Research Group, CHU de Lille

## Manuscript

IEEE conference PDF (≤6 pages): [`manuscript/ehpwas2026/`](manuscript/ehpwas2026/)

- Sources: `main.tex` / `main.pdf`
