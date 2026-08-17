# Point-by-point response to EHPWAS26 reviewers (ZKFL-PQ)

**Paper:** Zero-Knowledge Federated Learning with Lattice-Based Hybrid Encryption for Quantum-Resilient Medical AI  
**Venue:** eHPWAS 2026 / WiMob 2026 · PDF eXpress ID **61911X** · camera-ready ≤ 6 IEEE pages  
**Artifact:** https://github.com/edlansiaux/pq-zkfl-medical · branch `fix/ehpwas-binding` · tip **`1734a01`**

Status legend: **Done** = addressed in manuscript + code · **Partial** = measured / scoped honestly · **Deferred** = explicit residual (proof-engineering / scale)

---

## Reviewer A (Reject → addressed in camera-ready)

### A1. Experimental evaluation too limited (synthetic, 5 clients, one large-norm attack)
**Response (Done).** We added a multi-seed synthetic study (`experiments/run_baselines.py`, seeds 42–46) with FedAvg, ℓ₂-clipping, Multi-Krum, HE-only, ZKP-only, and hybrid, under **large-norm** and **sign-flip** attacks. We also run the **target protocol** on **UCI Breast Cancer** (`run_target_protocol.py`) and ship a **MedMNIST** path (`run_medmnist.py`).  
Manuscript: Sec. V; Tables of accuracy/detection; Fig. medical trajectory.

### A2. Cryptographic construction / BFV parameters insufficiently detailed
**Response (Done).** Camera-ready specifies language \(\mathcal{L}_\tau^{\mathrm{bind}}\), Σ-protocol (commit / FS–Unruh challenge / response / verify), BFV table \((n,q,t)\), quantization, noise growth, packing/chunking, averaging after threshold open, Enc-consistency gadget for coins \(\rho\), and HE presets (`classic128_demo` / `classic128`) with `lattice_security.py`.  
Manuscript: Sec. III–IV; Table of crypto parameters.

### A3. Complete formally specified ZKP (statement, witness, equations, FS, soundness/ZK, parameters)
**Response (Done).** Statement/witness, commitment equations, algebraic verify, Unruh \(r{=}128\), Lean~4 uniqueness theorem (`formal/lean`, `lake build`), Python game hops G0–G3, EasyCrypt sources.

### A4. Fully specify BFV (moduli, degree, encoding, quantization, noise, security estimate, packing, averaging)
**Response (Done).** See BFV parameter table; encode/decode scale; chunk packing for full \(d\); aggregate-then-threshold-decrypt mean; optional TenSEAL/SEAL backend (`ZKFL_HE_BACKEND=tenseal`). Security class documented via HomomorphicEncryption.org-style presets + reporter (estimator optional when installed).

### A5. Real public medical dataset + more clients + more rounds
**Response (Done).** UCI Breast Cancer; **PneumoniaMNIST full-resolution** (`run_medmnist_fullres.py`, 784-D, no projection); scale study **N=20, T=30** (`run_scale.py`).

### A6. Adaptive low-norm / model replacement / sign-flip / backdoors
**Response (Done).** Sign-flip measured; **backdoor trigger study** with FedAvg / ZKP / Krum / `hybrid_zkp_median` (`run_backdoor.py`). Pure ℓ₂-ZKP insufficient; complementary median defense closes the residual.

### A10. Clarify quantum-security boundary (classical ROM FS; QROM)
**Response (Done).** Unruh NIZK + **Lean 4** machine-checked uniqueness/2^{-r} (`formal/lean`) + Python QROM game hops + EasyCrypt sources.

### A7. Strong baselines and ablations (robust agg, clip, HE w/o ZKP, ZKP w/o HE, PQ transport)
**Response (Done).** Baselines: FedAvg, clip, Multi-Krum, HE-only, ZKP-only, hybrid (+ ML-KEM in hybrid path). Reported in Sec. V / `baseline_results.json`.

### A8. Repeated trials, uncertainty; explain anomalous 100% accuracy
**Response (Done).** Five seeds with mean±std. The former 100% synthetic accuracy is **not** treated as a medical claim; medical demo reports ~88.6%→93.9% (UCI). Hybrid large-norm detection ~97% mean with variance shown.

### A9. Measure communication and computation (eHPWAS/WiMob)
**Response (Done).** Per-round wall time and payload KB/MB for full-vector HE (~1.4–1.7 MB/round medical demo; synthetic timing ~20× FedAvg). Manuscript WiMob-oriented accounting.

### A10. Clarify quantum-security boundary (classical ROM FS; QROM)
**Response (Done).** Unruh NIZK + **Lean 4** machine-checked uniqueness/2^{-r} (`formal/lean`) + Python QROM game hops + EasyCrypt sources.

### A11. Threshold BFV or narrow privacy claims
**Response (Done).** \((t,n)=(2,3)\) **partial decryption** without reconstructing \(sk\); server \(sk=\bot\). Privacy claims match this model.

### A12. Reproducible artifact (versioned code, parameters, seeds, commands)
**Response (Done).** Public repo, pinned scripts, JSON results, `SECURITY.md`, `README`, seeds in configs. Tip commit cited above.

---

## Reviewer B (Accept if Room)

### B1. BFV \(n=512\) probably not 128-bit; give concrete levels
**Response (Done).** Presets: `classic128_demo` (\(n=512\), workshop CPU) and `classic128` (\(n=4096\)). Claimed class documented; optional TenSEAL path; estimator reporter. We do **not** over-claim SEAL-certified estimator output without the package.

### B2. ZKP only sketched; need parameters, proof sizes, soundness/ZK, Enc-consistency
**Response (Done).** Concrete dims, Unruh reps, proof size bytes in metrics; Enc-consistency Σ-protocol in `crypto/enc_consistency.py` bound into associated data with the norm proof.

### B3. FS only classical ROM; Unruh/Fischlin mitigation
**Response (Done).** Unruh parallel binary sessions implemented (`qrom_nizk.py`); default \(r{=}128\).

### B4. Partial HE (512/108996) breaks aggregator privacy
**Response (Done).** Full-vector chunking; all coordinates encrypted in target protocol.

### B5. Security levels of \((n,q,t,\mathrm{noise})\); avoid modular wrap in encoding/norm when summing clients
**Response (Done / Partial).** Parameters table + noise growth \(N\cdot O(\sigma\sqrt{n})\); demo recovers plaintext for \(N{=}5\). Wrap-around mitigated by scale/\(t\) choice and chunking; larger \(N\) needs RNS / SEAL path (stated).

### B6. Per-client / per-round payloads vs model dimension (full-vector packing)
**Response (Done).** Payload accounting and medical MB/round figures; scales with \(\lceil d/n\rceil\) chunks and Unruh \(r\).

### B7. Define Quantize; is \(\tau\) adaptive? Sensitivity to \(\tau\)
**Response (Done / Partial).** Quantize: clip \(c\), scale \(s\), round into \(\mathbb{Z}_t\) (defaults given). Fixed \(\tau\) in reported runs (synthetic 5; medical 8); adaptive rule noted as compatible. Ablation of \(\tau\) exists in legacy artifact tables; sensitivity discussion in manuscript.

### B8. Nontrivial real medical / vision results under benign + low-norm poisoning
**Response (Done).** UCI medical; **PneumoniaMNIST full-res 784-D** (no projection); sign-flip + backdoor studies.

---

## Reviewer C (Accept if Room)

### C1. Include additional results (column space available)
**Response (Done).** Six-page budget used for baselines (multi-seed, two attacks), medical demo, security analysis, Beskar positioning, payloads, and reproducibility pointers.

---

## Reviewer D (Accept if Room)

### D1. Code ≠ claims: ZKP on 256 dims, HE on 512 of 108996; no ct binding
**Response (Done).** Shared full dimension for prove+encrypt; Fiat–Shamir/Unruh digests ciphertext bytes (+ Enc-consistency first messages); encrypt-then-prove.

### D2. Malicious client can poison unproven coordinates
**Response (Done).** Closed by full-vector HE + bound proof on the same vector + Enc-consistency.

### D3. Trusted client Boolean `is_within_bound`
**Response (Done).** Acceptance = cryptographic verify only; Boolean not used.

### D4. 100% detection is for trivial oversized attack, not real soundness
**Response (Done / Partial).** Detection now via crypto verify; sign-flip study shows ℓ₂-only limits honestly. Trivial large-norm still used as positive control.

### D5. Server holds `sk`; plaintext full updates retained; threshold essential not optional
**Response (Done).** Threshold partial decrypt; server has no monolithic `sk`; aggregation on ciphertexts of accepted clients.

### D6. Eval too limited (synthetic 100%, 5 clients, 10 rounds, one attack); missing hardware, repeats, comms, library comparison
**Response (Done / Partial).** Multi-seed + medical + payloads + Unruh/HE timing. Hardware = laptop CPU NumPy (stated). Optional TenSEAL comparison path shipped.

### D7. Homemade crypto not FIPS/128-bit demonstration
**Response (Done).** ML-KEM-768 remains FIPS-203-oriented transport; HE claims scoped to Classic-128 **class** + SEAL backend option; no false “FIPS HE” claim.

### D8. Related work incomplete — discuss Beskar (“Efficient Full-Stack Private Federated Deep Learning With Post-Quantum Security”); fix refs [9]/[10]
**Response (Done).** Beskar table + explicit non-replacement positioning; bibliography cleaned in camera-ready.

---

## Conference production checklist (author)

| Requirement | Status |
|-------------|--------|
| ≤ 6 IEEE pages | Camera-ready `main.pdf` |
| PDF eXpress ID **61911X** | Use ieee-pdf-express.org |
| Copyright via EDAS | Author action |
| Author registration by **1 Sep 2026** | Author action |
| Upload CR with registration code | Author action |

---

## One-line summary for EDAS author response (optional paste)

> We addressed every actionable A–D remark in the camera-ready manuscript and public artifact (`cc3bd28`): full-vector HE with ciphertext-bound Unruh NIZK and Enc-consistency, threshold partial decryption, multi-seed baselines including sign-flip, scale N=20/T=30, backdoor evaluation with hybrid_zkp_median, full-res MedMNIST, Lean/EasyCrypt/Python QROM support, WiMob payloads, and Beskar positioning.
