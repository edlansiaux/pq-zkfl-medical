# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the code and experiments for the paper:

> **Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI**
> Edouard Lansiaux — CHU de Lille, Université de Lille

We propose **ZKFL-PQ**, a three-tiered cryptographic protocol for federated learning combining:

1. **ML-KEM-768** (FIPS 203) — Quantum-resistant key encapsulation based on Module-LWE
2. **Lattice-based Zero-Knowledge Proofs** — Verifiable gradient integrity via Σ-protocols with SIS-based commitments
3. **BFV Homomorphic Encryption** — Privacy-preserving gradient aggregation on ciphertexts

### Key Results

| Metric | Standard FL | FL + ML-KEM | **ZKFL-PQ (Ours)** |
|--------|-----------|------------|-------------------|
| Final Accuracy | 26.0% | 25.0% | **100.0%** |
| Byzantine Detection | 0% | 0% | **100%** |
| Quantum Resistant | ✗ | ✓ | **✓** |
| Gradient Privacy (vs. server) | ✗ | ✗ | **✓** |

## Repository Structure

```
pq-zkfl-medical/            
├── crypto/
│   ├── ml_kem.py             # ML-KEM-768 implementation (MLWE-based)
│   ├── zkp_norm.py           # ZKP for L2 norm bounds (lattice commitments)
│   └── homomorphic.py        # BFV homomorphic encryption
├── fl_core/
│   └── model.py              # MLP model + synthetic data + non-IID partitioning
├── experiments/
│   ├── run_experiment.py      # Main experiment runner (3 configurations)
│   └── plot_figures.py        # Publication figure generation
├── results/
│   └── experiment_results.json
├── figures/
│   ├── fig1_accuracy.pdf
│   ├── fig2_loss.pdf
│   ├── fig3_timing.pdf
│   ├── fig4_security_radar.pdf
│   ├── fig5_communication.pdf
│   └── fig6_breakdown.pdf
├── requirements.txt
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
# Run all three FL configurations (Standard, ML-KEM, Hybrid)
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
- Proves: ‖Δw‖₂ ≤ τ without revealing Δw

### BFV Homomorphic Encryption (`crypto/homomorphic.py`)
- Ring-LWE based scheme over Z_q[X]/(X^n + 1)
- Supports additive homomorphism for gradient aggregation
- Chunking for gradients exceeding polynomial degree

## Citation

```bibtex
@article{lansiaux2026zkflpq,
  title={Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption
         for Quantum-Resilient Medical AI},
  author={Lansiaux, Edouard},
  journal={arXiv preprint arXiv:2026.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Edouard Lansiaux** — [edouard.lansiaux@univ-lille.fr](mailto:edouard.lansiaux@univ-lille.fr)
- STaR-AI Research Group, CHU de Lille
