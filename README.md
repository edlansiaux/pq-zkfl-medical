# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PR](https://img.shields.io/badge/PR-fix%2Fehpwas--binding-blue)](https://github.com/edlansiaux/pq-zkfl-medical/pull/1)

## Overview

Code and camera-ready manuscript for:

> **Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical**  
> Edouard Lansiaux — eHPWAS 2026 / WiMob 2026

**ZKFL-PQ** combines:

1. **ML-KEM-768** (FIPS 203) — PQ key encapsulation
2. **Lattice norm ZKP** — Sigma-protocol + **Unruh-style** NIZK, bound to BFV ciphertext bytes
3. **BFV HE** — full-vector chunked aggregation with **(t,n)-threshold** decryption (no monolithic server `sk`)

Branch: [`fix/ehpwas-binding`](https://github.com/edlansiaux/pq-zkfl-medical/tree/fix/ehpwas-binding) · PR: [#1](https://github.com/edlansiaux/pq-zkfl-medical/pull/1)

## eHPWAS 2026 camera-ready

| Item | Path |
|------|------|
| 6-page IEEE PDF | [`manuscript/ehpwas2026/main.pdf`](manuscript/ehpwas2026/main.pdf) |
| LaTeX source | [`manuscript/ehpwas2026/main.tex`](manuscript/ehpwas2026/main.tex) |
| Build / PDF eXpress notes | [`manuscript/ehpwas2026/README.md`](manuscript/ehpwas2026/README.md) |
| Security / honesty notes | [`SECURITY.md`](SECURITY.md) |

Conference PDF eXpress ID: **61911X** · deadline **1 September 2026**.

## Quick start

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
git checkout fix/ehpwas-binding
pip install -r requirements.txt
```

Requires Python >= 3.9 and: numpy, scipy, matplotlib, cryptography, scikit-learn.

### Experiments

```bash
# Target protocol: UCI Breast Cancer + full-vector HE + threshold BFV + Unruh NIZK
python experiments/run_target_protocol.py

# Multi-baseline / multi-seed (FedAvg, clip, Multi-Krum, HE, ZKP, hybrid)
# Attacks: large_norm and sign_flip
python experiments/run_baselines.py

# Legacy three-config runner + ablations
python experiments/run_experiment.py
```

Results: `results/target_protocol_results.json`, `results/baseline_results.json`.

### Representative numbers (post-fix)

- **UCI Breast Cancer** target demo: final accuracy about **93.9%**, malicious client rejected, mean round about **4.5 s**, payload about **1.4-1.7 MB**/round.
- **Synthetic 5-seed baselines**: see camera-ready Table I and `results/baseline_results.json`. Sign-flip within tau yields **0%** L2 detection (honest limit vs Multi-Krum).

## Repository layout

```
pq-zkfl-medical/
  crypto/
    ml_kem.py           # ML-KEM-768
    zkp_norm.py         # Norm ZKP + ciphertext-bound FS
    qrom_nizk.py        # Unruh-style parallel NIZK
    homomorphic.py      # BFV + ThresholdBFV + full-vector manager
  fl_core/
    model.py            # MLP + synthetic + UCI Breast Cancer / MedMNIST
  experiments/
    run_target_protocol.py
    run_baselines.py
    run_experiment.py
    plot_figures.py
  results/
  manuscript/ehpwas2026/   # Camera-ready IEEE (6 pages)
  SECURITY.md
  requirements.txt
  README.md
```

## Security (summary)

| Feature | Status |
|---------|--------|
| FS / Unruh bound to HE ciphertext | Yes |
| Trusted client `is_within_bound` | Removed |
| Full-vector HE | Yes (all chunks) |
| Threshold decrypt `(2,3)` | Yes — server holds only `pk` |
| Real medical data | UCI Breast Cancer (optional MedMNIST) |
| ~128-bit HE | **Target class** only — demo `HE_N=512`; not SEAL/estimator-certified |
| Machine-checked QROM | Not claimed — Unruh transform is research-grade |

Details: [`SECURITY.md`](SECURITY.md).

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

MIT License. See [LICENSE](LICENSE).

## Contact

- **Edouard Lansiaux** — [edouard.lansiaux@orange.fr](mailto:edouard.lansiaux@orange.fr)
- STaR-AI and Emergency Department, CHU de Lille
