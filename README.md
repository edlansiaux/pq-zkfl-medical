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

### Key Results

| Metric | Standard FL | FL + ML-KEM | **ZKFL-PQ (Ours)** |
|--------|-------------|-------------|-------------------|
| Mean round time (s) | 0.149 | 2.376 | 2.912 |
| Final Accuracy | 23.0% | 23.5% | **100.0%** |
| Byzantine Detection | 0% | 0% | **100%** |
| Quantum Resistant | ✗ | ✓ | **✓** |
| Gradient Privacy (vs. server) | ✗ | ✗ | **✓** |

> **Note:** The ~20× overhead is compatible with clinical research workflows operating on daily/weekly training cycles.

## Security Notes

- **ZKP Verification**: The implementation includes full algebraic verification (`A·[z || r_z] ≡ T + c·C mod q`), ensuring SIS-based soundness.
- **Random Oracle Model**: Security proofs are in the classical ROM. QROM analysis remains future work.
- **Partial HE Coverage**: Only 512/108,996 parameters are HE-encrypted for computational tractability.

## Repository Structure

```
pq-zkfl-medical/            
├── crypto/
│   ├── ml_kem.py             # ML-KEM-768 implementation (MLWE-based)
│   ├── zkp_norm.py           # ZKP for L2 norm bounds (with algebraic verification)
│   └── homomorphic.py        # BFV homomorphic encryption
├── fl_core/
│   └── model.py              # MLP model + synthetic data + non-IID partitioning
├── experiments/
│   ├── run_experiment.py     # Main experiment runner (3 configurations + ablations)
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
- Supports additive homomorphism for gradient aggregation
- Parameters: n=512, q=2³²-5, t=2¹⁶
- Chunking for gradients exceeding polynomial degree

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

## Known Limitations

1. **Synthetic data only** — Validation on real medical imaging required
2. **Partial HE** — Only 512 params encrypted; full coverage would increase communication ~100×
3. **ℓ₂-norm only** — Does not prevent subtle low-norm or backdoor attacks
4. **Classical ROM** — QROM security analysis is future work

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
