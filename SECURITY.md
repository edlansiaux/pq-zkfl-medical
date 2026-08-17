# Security notes — pq-zkfl-medical (eHPWAS follow-up)

## Closed vs prior public tree

1. **ZKP ↔ HE binding** — Fiat–Shamir digests ciphertext bytes; no trusted `is_within_bound`.
2. **Shared / full-vector HE** — encrypts all parameter chunks (`GradientHEManager`, `HE_N` packing).
3. **Threshold BFV** — `(t,n)` Shamir shares of `sk`; server does **not** hold a monolithic secret (`use_threshold=True`).
4. **QROM-oriented NIZK** — `crypto/qrom_nizk.py` Unruh-style parallel binary sessions with invertible-RO records.
5. **Real medical data** — UCI Breast Cancer Wisconsin via `load_medical_dataset("breast_cancer")` (optional MedMNIST).

## Parameter honesty

| Claim | Reality |
|-------|---------|
| ~128-bit HE | `HE_CLAIMED_SECURITY_BITS=128` is a **target class** (HE.org Classic-128 often uses `n=4096`). This NumPy encoder uses smaller `HE_N` for CPU demos and is **not** a lattice-estimator certificate / SEAL drop-in. |
| Unruh QROM | Transform implemented; **not** a machine-checked QROM proof. Increase `unruh_reps` for higher soundness. |
| Threshold | Reconstructs `sk` from `t` shares then decrypts (honest-majority offline style). Production would use distributed decryption without reconstructing `sk`. |
| Aggregator privacy | Holds only with threshold decryptors + full-vector HE under the stated trust model. |

## Commands

```bash
pip install -r requirements.txt   # includes scikit-learn
python experiments/run_target_protocol.py
python experiments/run_baselines.py
```
