# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the code and experiments for the paper:

> **Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI**  
> Edouard Lansiaux

We propose **ZKFL-PQ**, a three-tiered cryptographic protocol for federated learning combining:

1. **ML-KEM-768** (FIPS 203) — Quantum-resistant key encapsulation based on Module-LWE
2. **Lattice-based Zero-Knowledge Proofs** — Verifiable gradient integrity via Σ-protocols with SIS-based commitments and **full algebraic verification**
3. **BFV Homomorphic Encryption** — Privacy-preserving gradient aggregation on ciphertexts

### Key Results (target protocol / UCI Breast Cancer)

| Metric | Value |
|--------|-------|
| Final accuracy (3 rounds) | **93.9%** |
| Large-norm detection (attack round) | **yes** |
| Payload / round | ≈ 1.4–1.7 MB (full-vector HE) |
| Unruh default | **r=128** (demos often 32–64) |
| Threshold | **(2,3) partial decrypt** (`sk` never assembled) |

See [`SECURITY.md`](SECURITY.md) for the closed vs residual checklist. Re-run:

```bash
python experiments/run_target_protocol.py
python experiments/run_baselines.py
python experiments/run_medmnist.py   # needs medmnist (falls back to UCI)
```

## Security Notes

- **ZKP Verification**: Algebraic check `A·[z || r_z] ≡ T + c·C mod q`, plus Fiat–Shamir / Unruh binding to BFV ciphertext bytes (`associated_data`).
- **No trusted Boolean**: acceptance does **not** use client-supplied `is_within_bound`.
- **Full-vector HE**: all coordinates encrypted (chunked); ZKP covers the same vector.
- **Threshold BFV**: server `sk=None` after share generation; open via Lagrange-weighted partial decryptions.
- **Enc-consistency**: `crypto/enc_consistency.py` proves knowledge of BFV coins ρ for each ciphertext.
- **SEAL backend**: `ZKFL_HE_BACKEND=tenseal` (requires `pip install tenseal`).
- **QROM-oriented Unruh**: default `r=128`; combinatorial soundness machine-checked in `formal/check_unruh_soundness.py`.
- **HE presets**: `ZKFL_HE_PRESET=classic128_demo` (n=512) or `classic128` (n=4096); see `crypto/lattice_security.py`.
- See [`SECURITY.md`](SECURITY.md) for the eHPWAS review remediation checklist.

## Repository Structure

```
pq-zkfl-medical/            
├── crypto/
│   ├── ml_kem.py             # ML-KEM-768 implementation (MLWE-based)
│   ├── zkp_norm.py           # ZKP for L2 norm bounds (with algebraic verification)
│   ├── qrom_nizk.py          # Unruh-style QROM-oriented NIZK
│   ├── lattice_security.py   # HE preset + optional estimator report
│   └── homomorphic.py        # BFV + ThresholdBFV (partial decrypt)
├── fl_core/
│   └── model.py              # MLP + synthetic / UCI / MedMNIST loaders
├── experiments/
│   ├── run_experiment.py     # Main experiment runner (3 configurations + ablations)
│   ├── run_baselines.py      # Multi-seed FedAvg / clip / Krum / hybrid
│   ├── run_target_protocol.py
│   ├── run_medmnist.py
│   └── plot_figures.py       # Publication figure generation
├── results/
│   └── experiment_results.json
├── figures/
│   ├── fig1_accuracy.pdf     # Accuracy convergence
│   ├── fig2_loss.pdf         # Loss convergence
│   ├── fig3_timing.pdf       # Timing comparison
│   ├── fig4_security_radar.pdf
│   ├── fig5_communication.pdf
│   ├── fig6_breakdown.pdf    # ZKFL-PQ component breakdown
│   ├── fig7_ablation_malicious.pdf
│   └── fig8_ablation_threshold.pdf
├── manuscript/
│   └── main.tex              # LaTeX source
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Requirements

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, cryptography

### Installation

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run all three FL configurations + ablation studies
python experiments/run_experiment.py

# Generate publication figures
python experiments/plot_figures.py
```

### Compile Manuscript

```bash
cd manuscript
pdflatex main.tex && pdflatex main.tex  # Two passes for references
```

## Cryptographic Implementations

### ML-KEM-768 (`crypto/ml_kem.py`)
- Simplified but mathematically faithful implementation of FIPS 203
- Parameters: n=256, k=3, q=3329, η₁=η₂=2
- Includes KeyGen, Encaps, Decaps + AES-256-CTR symmetric layer

### ZKP for Norm Bounds (`crypto/zkp_norm.py`)
- Σ-protocol with Fiat-Shamir transform for non-interactivity
- SIS-based lattice commitments (post-quantum binding)
- Rejection sampling for zero-knowledge property
- **Full algebraic verification**: `A·[z || r_z] ≡ T + c·C (mod q)`
- Proves: ‖Δw‖₂ ≤ τ without revealing Δw

### BFV Homomorphic Encryption (`crypto/homomorphic.py`)
- Ring-LWE based scheme over Z_q[X]/(X^n + 1)
- Full-vector chunking + `(t,n)` **ThresholdBFV** partial decryption (no reconstructed `sk`)
- Presets via `ZKFL_HE_PRESET`: `classic128_demo` (n=512) / `classic128` (n=4096)

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

## Known Limitations (narrow)

1. **ℓ₂-norm alone** — Insufficient against sign-flip/backdoors; compose with `hybrid_zkp_median` / Krum (evaluated in `run_backdoor.py`)
2. **SEAL ⊕ threshold** — TenSEAL path and NumPy threshold path are both shipped; fusing them in one process is optional engineering
3. **EasyCrypt stdlib QROM** — Lean 4 + Python game hops machine-checked; EasyCrypt sources included for `easycrypt` users

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

- **Edouard Lansiaux** — [edouard.lansiaux@orange.fr](mailto:edouard.lansiaux@orange.fr)
- STaR-AI Research Group, CHU de Lille

## eHPWAS 2026 camera-ready
IEEE workshop manuscript (6 pages): [\`manuscript/ehpwas2026/\`](manuscript/ehpwas2026/)
- PDF: `manuscript/ehpwas2026/main.pdf`
- Branch: `fix/ehpwas-binding`
