# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)](requirements.txt)

**Paper:** *Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI* — Edouard Lansiaux  
**Artifact:** this repository (`main`) · Manuscript: [`manuscript/ehpwas2026/`](manuscript/ehpwas2026/)  
**Contact:** [edouard1.lansiaux@chu-lille.fr](mailto:edouard1.lansiaux@chu-lille.fr) · STaR-AI / Emergency Department, CHU de Lille

---

## What this repository is

**ZKFL-PQ** is a *layered* protocol for medical federated learning under Harvest-Now-Decrypt-Later (HNDL), Byzantine poisoning, and curious-aggregator threats. It is **not** a drop-in replacement for Beskar-class full-stack secure aggregation; it complements such stacks with ciphertext-bound norm proofs, multi-party decrypt, and post-ZKP robust aggregation.

| Layer | Role | Default in artifact |
|-------|------|---------------------|
| 1. ML-KEM-768 | PQ transport (FIPS 203 L3) | `crypto/ml_kem.py` |
| 2. Unruh NIZK + Enc-consistency | Prove \(\|\tilde w\|_2\le\tau\) bound to BFV `ct` + coins \(\rho\) | `r=128` library default; demos often `4–64` |
| 3. Full-vector BFV + \((t,n)\) threshold | Encrypt all \(d\) coords; open via **partial** decrypt (`sk` never rebuilt) | `(2,3)`; fused SEAL+threshold if TenSEAL installed |
| 4. Post-ZKP robust agg | Median (default) / Multi-Krum / mean | `ZKFL_ROBUST_AGG=median` |

**Threat coverage (composition matters):**

| Threat | ML-KEM | ZKP \(\ell_2\) | Enc-cons. | Thr. open | Median |
|--------|--------|----------------|-----------|-----------|--------|
| HNDL / PQ transit | yes | — | — | — | — |
| Oversized update | — | yes | — | — | — |
| Proof/ct split | — | bind | yes | — | — |
| Curious aggregator | — | — | — | yes | — |
| Sign-flip / in-bound backdoor | — | no (in-bound) | — | — | yes |

---

## Key empirical numbers (from `results/*.json`)

| Study | Result |
|-------|--------|
| Synthetic Hybrid+median (5 seeds) | Large-norm & sign-flip final acc **100%**; crypto det. large-norm **≈97%**, sign-flip **0%** (must accept in-bound) |
| UCI Breast Cancer target stack | Acc **88.6% → 92.1% → 92.1%** (3 rounds); oversized client rejected on attack round; fused HE + median + Unruh `r=64` |
| Backdoor ASR | FedAvg / \(\ell_2\) ≈**98%** → ZKP+median ≈**55%** (clean ≈**99%**); Multi-Krum ≈**46%** |
| ConvNet28 PneumoniaMNIST smoke | ≈**93.3%** acc, oversized rejected; `d≈51618`, HE chunks **101**, Unruh `r=4` |
| Microbench (shared SIS + modular BFV, `d≈51.6k`) | Unruh prove/verify ≈**0.10 / 0.06 s**; HE encrypt ≈**3.5 s** |

Reproduce tables/figures from the scripts below; do not treat synthetic 100% accuracy as a clinical claim.

---

## Quick start

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
pip install -r requirements.txt
pip install tenseal medmnist   # optional: fused SEAL path + MedMNIST

python experiments/smoke_residuals.py
python experiments/run_target_protocol.py   # fused HE + median by default
python formal/run_formal_ci.py              # Lean/Python/EasyCrypt surface
```

**Environment overrides:**

| Variable | Values | Meaning |
|----------|--------|---------|
| `ZKFL_HE_BACKEND` | `fused` (default if TenSEAL), `numpy`, `tenseal` | HE manager |
| `ZKFL_HE_PRESET` | `classic128_demo` (`n=512`, default), `classic128` (`n=4096`) | BFV degree / moduli class |
| `ZKFL_ROBUST_AGG` | `median` (default), `krum`, `mean` | Post-ZKP aggregation |
| `ZKFL_UNRUH_REPS` | int (CNN default often `4`) | Parallel Unruh sessions |
| `ZKFL_CNN_SAMPLES` / `ZKFL_CNN_CLIENTS` / `ZKFL_CNN_ROUNDS` | ints | ConvNet28 smoke knobs |

---

## Repository layout (canonical `main`)

```
pq-zkfl-medical/
├── crypto/
│   ├── ml_kem.py              # ML-KEM-768 + AES-CTR wrap
│   ├── zkp_norm.py            # Σ-protocol, SIS Commit, crypto-only accept
│   ├── qrom_nizk.py           # Unruh NIZK; ONE shared LatticeCommitment
│   ├── enc_consistency.py     # BFV coins Σ-gadget
│   ├── homomorphic.py         # BFV + ThresholdBFV partial decrypt + parallel chunk enc
│   ├── fused_he.py            # FusedSealThresholdHE (NumPy threshold + SEAL sidecar)
│   ├── seal_backend.py        # Optional TenSEAL path
│   ├── keccak_f1600.py        # Bit-level Keccak-f[1600] + SHA3-256 (FIPS match)
│   └── lattice_security.py    # HE presets / estimator notes
├── fl_core/
│   ├── model.py               # MLP + synthetic / UCI / MedMNIST loaders
│   ├── cnn.py                 # ConvNet28 (~51.6k params, no compact head)
│   └── robust_agg.py          # median / Multi-Krum / mean
├── experiments/
│   ├── run_target_protocol.py # Full medical stack (UCI)
│   ├── run_baselines.py       # Multi-seed FedAvg / clip / Krum / hybrid
│   ├── run_backdoor.py        # Trigger ASR (+ hybrid_zkp_median)
│   ├── run_scale.py           # N=20, T=30
│   ├── run_medmnist.py
│   ├── run_medmnist_fullres.py
│   ├── run_medmnist_cnn.py    # ConvNet28 + HE + Unruh
│   ├── run_experiment.py      # Shared FL loop helpers
│   ├── smoke_residuals.py
│   ├── plot_figures.py
│   └── make_excellence_figure.py
├── formal/
│   ├── run_formal_ci.py       # One-shot formal gate
│   ├── check_keccak_bitlevel.py   # FIPS + algebraic lane lemmas
│   ├── check_sha3_qrom.py
│   ├── check_unruh_soundness.py
│   ├── check_unruh_qrom_games.py
│   ├── lean/                  # Combinatorial Unruh 2^{-r}
│   └── easycrypt/
│       ├── lib/KeccakF1600.ec # θ/ρ/π/χ/ι + lane lemmas
│       ├── lib/SHA3.ec, QROMCore.ec
│       ├── QROM.ec, SHA3_QROM.ec
│       └── UnruhBinaryQROM.ec, UnruhBinaryCounting.ec
├── manuscript/ehpwas2026/     # IEEE ≤6-page sources + PDF + figures
├── results/                   # JSON logs (source of truth for reported numbers)
├── SECURITY.md
├── requirements.txt
├── LICENSE
└── README.md                  # this file
```

**Canonical branch:** `main` only. Do not use stale feature branches.

---

## Cryptographic design (implementation notes)

### ML-KEM-768 (`crypto/ml_kem.py`)
- Module-LWE oriented parameters: \(n=256\), \(k=3\), \(q=3329\) (FIPS 203 Level 3 class).
- KeyGen / Encaps / Decaps; payload wrap with AES-256-CTR under the shared secret.

### Norm ZKP + Unruh (`crypto/zkp_norm.py`, `crypto/qrom_nizk.py`)
- SIS-style \(\mathrm{Commit}(m;r)=\mathbf{A}[m\|r]\bmod q\) with algebraic verify \(\mathbf{A}[z\|r_z]\equiv T+cC\).
- **Acceptance never uses a client Boolean** (`is_within_bound` is ignored).
- Unruh: \(r\) parallel binary sessions with invertible RO records; challenges digest BFV ciphertext bytes.
- **Performance:** one shared \(\mathbf{A}\) of shape \(256\times(d+128)\) for all sessions (and the base Σ object), plus batched verify matmul — critical at \(d\sim 5\times 10^4\).

### Enc-consistency (`crypto/enc_consistency.py`)
- Proves knowledge of coins \(\rho=(u,e_0,e_1)\) for each BFV chunk so a benign proof cannot attach to a mismatched ciphertext.

### BFV + threshold (`crypto/homomorphic.py`, `crypto/fused_he.py`)
- Full-vector chunking: \(\lceil d/n\rceil\) ciphertexts; all coordinates encrypted.
- After keygen, Shamir-share `sk` and erase the monolithic key; open with Lagrange-weighted **partial** decryptions + smudging.
- Modular exact negacyclic mul; parallel per-chunk encrypt with independent RNG streams.
- Default HE path if TenSEAL present: `FusedSealThresholdHE` (threshold privacy + SEAL sidecar consistency).

### SHA3 / Keccak formal surface (`crypto/keccak_f1600.py`, `formal/`)
- Executable Keccak-\(f\)[1600] matches `hashlib.sha3_256` (FIPS empty vector included).
- EasyCrypt `KeccakF1600.ec`: algebraic lane lemmas (θ-column, ρ, π bijection, χ local, ι locality, packing, 24-fold).
- CI flag strings: `MACHINE_CHECKED_KECCAK_BITLEVEL=1`, `MACHINE_CHECKED_KECCAK_LANE_LEMMAS=1`, `FORMAL_CI_OK=1`.

---

## Experiments (how to reproduce)

```bash
# 1) Smoke invariants (binding, crypto-only accept, threshold path)
python experiments/smoke_residuals.py

# 2) UCI target protocol (fused + median + Unruh r=64 in script defaults)
python experiments/run_target_protocol.py
# → results/target_protocol_results.json

# 3) Multi-seed synthetic baselines (large-norm + sign-flip)
python experiments/run_baselines.py
# → results/baseline_results.json

# 4) Backdoor ASR
python experiments/run_backdoor.py
# → results/backdoor_results.json ; optional figure helper:
python experiments/make_excellence_figure.py

# 5) Scale N=20, T=30
python experiments/run_scale.py

# 6) MedMNIST (needs: pip install medmnist)
python experiments/run_medmnist.py
python experiments/run_medmnist_fullres.py
python experiments/run_medmnist_cnn.py      # ConvNet28, no compact head

# 7) Formal CI
python formal/run_formal_ci.py
```

### Ablations (historical / residual)

Varying malicious clients (0–3) and \(\tau\) on the synthetic residual harness:

| # Malicious | Final Acc | Detection |
|-------------|-----------|-----------|
| 0–3 | 100% | 100% when malicious present |

| \(\tau\) | Detection | FP |
|----------|-----------|----|
| 1.0–2.0 | 100% | 13.6% |
| 5.0–50.0 | 100% | 0% |

---

## Manuscript

```bash
cd manuscript/ehpwas2026
# tectonic main.tex   # or pdflatex ×2
```

- Sources: `main.tex`, `main.pdf`, `figures/*.png`
- IEEE conference format, **≤ 6 pages**, PDF eXpress conference ID **61911X**
- Figures are PNG (avoid Type-3 fonts from some matplotlib PDF backends)

Authoritative numeric claims in the PDF should match `results/*.json` in this tree.

---

## Security & honesty bounds

See [`SECURITY.md`](SECURITY.md) for the closed residual checklist.

**We claim:** PQ transport; oversized-update soundness under SIS+Unruh/RO; ct binding + Enc-consistency; threshold privacy against aggregator + \(t-1\) decryptors; empirical median benefit vs in-bound attacks.

**We do not claim:** differential privacy; SEAL-certified Classic-128 for the NumPy `n=512` demo preset; a fully discharged EasyCrypt proof of every Keccak round identity without the executable checker; Beskar replacement.

---

## Citation

```bibtex
@article{lansiaux2026zkflpq,
  title={Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI},
  author={Lansiaux, Edouard},
  journal={arXiv preprint arXiv:2603.03398},
  year={2026},
  url={https://github.com/edlansiaux/pq-zkfl-medical}
}
```

## License

MIT — see [LICENSE](LICENSE).
