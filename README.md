# ZKFL-PQ: Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI

[![arXiv](https://img.shields.io/badge/arXiv-2603.03398-b31b1b.svg)](https://arxiv.org/abs/2603.03398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.9-blue.svg)](requirements.txt)

**Paper:** *Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI* — Edouard Lansiaux  
**Artifact branch:** `main` only · Manuscript: [`manuscript/ehpwas2026/`](manuscript/ehpwas2026/) (IEEE ≤6 pages, PDF eXpress **61911X**)  
**Contact:** [edouard1.lansiaux@chu-lille.fr](mailto:edouard1.lansiaux@chu-lille.fr) · STaR-AI / Emergency Department, CHU de Lille

---

## What this repository is

**ZKFL-PQ** is a *compositional* protocol for medical federated learning under HNDL, Byzantine poisoning, and curious-aggregator threats. It is **not** a Beskar replacement; it complements secure-aggregation stacks with ciphertext-bound dual-norm proofs, Unruh Enc-consistency, threshold open accountability, and post-ZKP robust aggregation.

| Layer | Role | Artifact default |
|-------|------|------------------|
| 1. ML-KEM-768 | PQ transport (FIPS 203 L3) | `crypto/ml_kem.py` |
| 2. Dual-norm gate | Unruh \(\ell_2\) on \(\mathcal{L}_{\tau_2,\tau_\infty}^{\mathrm{bind}}\) + public \(\ell_\infty\) clip on the **same** BFV plaintext | `qrom_nizk.py` + `fl_core/clip.py` · library \(r{=}128\) |
| 3. Unruh Enc-consistency | Coins \(\rho\) under the same Unruh transform as the norm proof | `unruh_enc_consistency.py` |
| 4. Full-vector BFV + \((t,n)\) open | All \(d\) coords; partial decrypt; **PartialDecrypt NIZK** aborts bad \(\mu_i\) | `(2,3)` · `homomorphic.py` + `partial_dec_nizk.py` |
| 5. Adaptive \(\tau_2\) + transcript | Public schedule + \(H_{t-1}\) in AD | `adaptive_tau.py` + `round_transcript.py` |
| 6. Post-ZKP robust agg | Median (default) / Multi-Krum / mean | `ZKFL_ROBUST_AGG=median` |

**Threat coverage (matches the paper table):**

| Threat | ML-KEM | Dual-norm | Unruh-Enc | PD-NIZK | Median |
|--------|--------|-----------|-----------|---------|--------|
| HNDL / PQ transit | yes | — | — | — | — |
| Oversized update | — | \(\ell_2\) | — | — | — |
| Sparse \(\ell_\infty\) spike | — | \(\ell_\infty\) | — | — | help |
| Proof/ct split | — | bind | yes | — | — |
| Malicious partial \(\mu_i\) | — | — | — | abort | — |
| Curious aggregator | — | — | — | thr. | — |
| Sign-flip / diffuse backdoor | — | no† | — | — | yes |

†In-bound updates may still satisfy \((\tau_2,\tau_\infty)\); crypto detection of sign-flip under \(\ell_2\) alone is 0% (see baselines).

---

## Parameter honesty (read before citing bits)

| Knob | Library / production-oriented | Latency demos (JSON) |
|------|-------------------------------|----------------------|
| Norm Unruh \(r\) | **128** | Innovation pack often **16**; CNN smoke often **4** |
| Enc-Unruh \(r_{\mathrm{enc}}\) | parallel / chunk | Innovation pack often **4** |
| HE preset | `classic128_demo` \(n{=}512\) (default) or `classic128` \(n{=}4096\) | Demo \(n{=}512\) is **not** a HomomorphicEncryption.org certificate |
| PartialDecrypt NIZK | Classical FS (ROM) | Not Unruh-lifted yet |

---

## Innovation pack (measured)

```bash
python experiments/run_innovation_pack.py
# → results/innovation_pack_results.json
```

| Result | Value |
|--------|-------|
| UCI pipeline (3 rounds) | **89.5% → 93.0% → 92.1%**; oversized rejected; \(\tau\) **8→2** |
| Dual-norm sparse poison | Reject **100%** vs \(\ell_2\)-only **0%**; ~**265×** lower coord spike |
| Tampered partial \(\mu_i\) | **Rejected** by PartialDecrypt NIZK |
| Wall-clock (UCI pack) | ≈**20–23 s**/round on laptop CPU |

Treat UCI/MedMNIST as **pipeline validation**, not clinical efficacy.

---

## Other key numbers (`results/*.json`)

| Study | Result |
|-------|--------|
| Synthetic Hybrid+median (5 seeds) | Large-norm & sign-flip final acc **100%**; crypto det. large-norm **≈97%**, sign-flip **0%** |
| UCI legacy target (`run_target_protocol.py`) | **88.6% → 92.1% → 92.1%**; fused HE + median + Unruh `r=64` |
| Backdoor ASR | FedAvg / \(\ell_2\) ≈**98%** → ZKP+median ≈**55%** (clean ≈**99%**); Multi-Krum ≈**46%** |
| ConvNet28 PneumoniaMNIST smoke | ≈**93.3%**; \(d≈51618\); Unruh `r=4` |
| Microbench \(d≈51.6\)k | Unruh prove/verify ≈**0.10 / 0.06 s**; HE encrypt ≈**3.5 s** |

---

## Quick start

```bash
git clone https://github.com/edlansiaux/pq-zkfl-medical.git
cd pq-zkfl-medical
pip install -r requirements.txt
pip install tenseal medmnist   # optional

python experiments/smoke_residuals.py
python experiments/run_innovation_pack.py
python experiments/run_target_protocol.py
python formal/run_formal_ci.py
```

| Variable | Values | Meaning |
|----------|--------|---------|
| `ZKFL_HE_BACKEND` | `fused` (default if TenSEAL), `numpy`, `tenseal` | HE manager |
| `ZKFL_HE_PRESET` | `classic128_demo` (`n=512`), `classic128` (`n=4096`) | BFV degree class |
| `ZKFL_ROBUST_AGG` | `median`, `krum`, `mean` | Post-ZKP aggregation |
| `ZKFL_UNRUH_REPS` | int | Parallel Unruh sessions (CNN often `4`) |

---

## Repository layout

```
pq-zkfl-medical/                 # branch: main only
├── crypto/                      # ML-KEM, Unruh, Enc, HE, Keccak, PD-NIZK, transcript
├── fl_core/                     # MLP, ConvNet28, clip, adaptive τ, robust agg
├── experiments/                 # target, innovation pack, baselines, backdoor, MedMNIST, formal smokes
├── formal/                      # Lean + EasyCrypt + Python CI gates
├── manuscript/ehpwas2026/       # CANONICAL camera-ready: main.tex, main.pdf, figures/*.png
├── figures/                     # Optional/legacy plot outputs (PNG); paper does not read this tree
├── results/                     # JSON source of truth for reported numbers
├── SECURITY.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

**Figures:** camera-ready PNGs live only under `manuscript/ehpwas2026/figures/` (paths in `main.tex`). Root `figures/` may hold experiment plot regenerations; do not mix the two when editing the PDF.

---

## Cryptographic notes (short)

- **Accept is crypto-only** — client Booleans are ignored.
- **Shared SIS \(\mathbf{A}\)** — one \(256\times(d+128)\) matrix for all Unruh sessions (required at ConvNet28 scale).
- **Unruh Enc-consistency** — same transform as norm Unruh; binds coins \(\rho\) to `ct`.
- **PartialDecrypt NIZK** — FS relation \(\mu = c_1 \star s_{\mathrm{eff}}\); abort on tamper (ROM).
- **Dual-norm** — \(\ell_2\) via Unruh; \(\ell_\infty\) via public clip + Enc-consistency binding (not a separate Unruh range proof).
- **Formal CI** — `MACHINE_CHECKED_KECCAK_BITLEVEL=1`, `MACHINE_CHECKED_KECCAK_LANE_LEMMAS=1`, `FORMAL_CI_OK=1`.

---

## Reproduce experiments

```bash
python experiments/smoke_residuals.py
python experiments/run_innovation_pack.py      # → innovation_pack_results.json
python experiments/run_target_protocol.py      # → target_protocol_results.json
python experiments/run_baselines.py            # → baseline_results.json
python experiments/run_backdoor.py             # → backdoor_results.json
python experiments/make_excellence_figure.py   # backdoor ASR figure helper
python experiments/run_scale.py
python experiments/run_medmnist_cnn.py         # needs medmnist
python formal/run_formal_ci.py
```

### Manuscript build

```bash
cd manuscript/ehpwas2026
tectonic main.tex   # or pdflatex ×2
```

Numeric claims in `main.pdf` must match `results/*.json`.

---

## Security & honesty bounds

See [`SECURITY.md`](SECURITY.md).

**We claim:** PQ transport; dual-norm membership under stated assumptions; Unruh Enc-consistency against proof/ct splits; threshold privacy + open abort on bad partials; transcript binding; empirical median benefit vs in-bound attacks; measured dual-norm sparse-poison reject rates.

**We do not claim:** differential privacy; clinical efficacy from UCI/MedMNIST; Classic-128 certification of NumPy \(n{=}512\); fully discharged EasyCrypt proof of every Keccak identity without the FIPS checker; Unruh/QROM for PartialDecrypt (still ROM-FS); Beskar replacement; that median eliminates backdoors (ASR ≈55% remains).

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
