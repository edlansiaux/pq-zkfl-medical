# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Code and experiments for:

> **Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI**  
> Edouard Lansiaux · arXiv:[2603.03398](https://arxiv.org/abs/2603.03398)

**ZKFL-PQ** combines:

1. **ML-KEM-768** (FIPS 203) — PQ transport  
2. **Lattice ZKP / Unruh NIZK** — ℓ₂-norm soundness bound to the BFV ciphertext (+ Enc-consistency for coins ρ)  
3. **BFV HE** — full-vector aggregation with `(t,n)` **partial** threshold decrypt (no reconstructed `sk`)

Camera-ready for **eHPWAS 2026 / WiMob 2026** (≤6 IEEE pages, PDF eXpress **61911X**): [`manuscript/ehpwas2026/`](manuscript/ehpwas2026/) · branch [`fix/ehpwas-binding`](https://github.com/edlansiaux/pq-zkfl-medical/tree/fix/ehpwas-binding).

Point-by-point reviewer replies: [`REVIEWER_RESPONSE.md`](REVIEWER_RESPONSE.md) · closed-residuals checklist: [`SECURITY.md`](SECURITY.md).

### Key results (artifact)

| Study | Highlight |
|-------|-----------|
| Target protocol (UCI Breast Cancer) | Acc ≈ **93.9%**; malicious large-norm rejected; ≈1.4–1.7 MB/round |
| Scale (`run_scale.py`) | **N=20, T=30** multi-method |
| Backdoor (`run_backdoor.py`) | Pure ℓ₂-ZKP insufficient; **`hybrid_zkp_median`** cuts ASR vs FedAvg |
| Full-res MedMNIST | **784-D**, no projection; ≈67.5% @ r3 + reject oversized client |
| Unruh | Default **r=128**; Lean 4 + Python game hops machine-checked |

## Quick start

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
git checkout fix/ehpwas-binding
pip install -r requirements.txt
# optional:
pip install tenseal medmnist
```

```bash
python experiments/smoke_residuals.py
python experiments/run_target_protocol.py
python experiments/run_baselines.py
python experiments/run_scale.py
python experiments/run_backdoor.py
python experiments/run_medmnist.py              # may fall back to UCI
python experiments/run_medmnist_fullres.py      # 784-D PneumoniaMNIST

python -m formal.check_unruh_soundness
python -m formal.check_unruh_qrom_games
cd formal/lean && lake build                    # needs Lean 4 / elan
```

Env knobs: `ZKFL_HE_PRESET=classic128_demo|classic128`, `ZKFL_HE_BACKEND=numpy|tenseal`, `ZKFL_DATASET=pneumoniamnist`.

## Security notes

- Crypto-only acceptance (no client `is_within_bound`)
- Full-vector HE + ZKP on the same vector; FS/Unruh digest ciphertext (+ Enc-consistency)
- Threshold BFV: server `sk=None`; Lagrange-weighted partial decrypt
- SEAL path via TenSEAL (`ZKFL_HE_BACKEND=tenseal`)
- Details: [`SECURITY.md`](SECURITY.md)

## Repository layout

```
crypto/           ml_kem, zkp_norm, qrom_nizk, enc_consistency, seal_backend,
                  homomorphic (ThresholdBFV), lattice_security
fl_core/          MLP + synthetic / UCI / MedMNIST loaders
experiments/      run_target_protocol, run_baselines, run_scale, run_backdoor,
                  run_medmnist, run_medmnist_fullres, smoke_residuals, …
formal/           Lean 4 Unruh lib, EasyCrypt sources, Python QROM game hops
manuscript/ehpwas2026/   camera-ready main.tex + main.pdf
results/          JSON outputs (target, baselines, scale, backdoor, fullres)
REVIEWER_RESPONSE.md
SECURITY.md
```

## Cryptographic modules

| Module | Role |
|--------|------|
| `crypto/ml_kem.py` | ML-KEM-768 + AES-CTR session layer |
| `crypto/zkp_norm.py` | Σ-protocol + FS; algebraic verify |
| `crypto/qrom_nizk.py` | Unruh NIZK (default r=128) |
| `crypto/enc_consistency.py` | Σ-gadget for BFV coins ρ |
| `crypto/homomorphic.py` | BFV + ThresholdBFV partial decrypt |
| `crypto/seal_backend.py` | Optional Microsoft SEAL (TenSEAL) |
| `formal/lean` | Machine-checked Unruh combinatorial lemmas (`lake build`) |

## Known limitations (narrow)

1. **ℓ₂ alone** does not stop sign-flip/backdoors — compose with `hybrid_zkp_median` / Krum (evaluated).  
2. **SEAL ⊕ threshold** ship as two paths; fusing in one process is optional.  
3. EasyCrypt **stdlib QROM linking** is for `easycrypt` users; Lean + Python hops are CI-checkable without it.

## Citation

```bibtex
@article{lansiaux2026zkflpq,
  title={Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI},
  author={Lansiaux, Edouard},
  journal={arXiv preprint arXiv:2603.03398},
  year={2026}
}
```

## License / contact

MIT — see [LICENSE](LICENSE).  
Edouard Lansiaux — STaR-AI, CHU de Lille — edouard.lansiaux@orange.fr
