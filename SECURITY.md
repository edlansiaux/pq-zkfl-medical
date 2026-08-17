# Security fixes (eHPWAS 2026 camera-ready follow-up)

## Changes

1. **`crypto/zkp_norm.py`**
   - Fiat–Shamir challenge digests `associated_data` (BFV ciphertext bytes).
   - Removed `is_within_bound` / `proof["accepted"]` from the proof acceptance path.
   - `verify_proof` is cryptographic only (norm / challenge / algebraic).
   - Rejection bound scaled for quantized integers (`QUANT_SCALE`).
   - Dim mismatch between prove vector and `self.dim` raises (no silent truncate).

2. **`experiments/run_experiment.py`**
   - `PROTECTED_DIM = min(HE_N, n_params)` shared by ZKP **and** HE (no 256/512 split).
   - Encrypt first, then prove with `associated_data=he_cts`.
   - Accept iff `verify_proof(...)` — no trusted Boolean.
   - Global update writes **only** protected coordinates from HE decrypt; unprotected coords stay unchanged (blocks suffix poisoning via plaintext `mean(valid_deltas)`).

3. **`crypto/homomorphic.py`**
   - Comment clarified: toy `n=512` is **not** a 128-bit HE claim.

## Still out of scope

- Full-vector HE / threshold decryption (server still holds `sk`).
- QROM-tight Fiat–Shamir.
- Real medical datasets / low-norm attacks.

## Push to GitHub

```powershell
cd C:\Users\edlsx\zkfl-ehwasp2026\repo-pq-zkfl
git init
git remote add origin https://github.com/edlansiaux/pq-zkfl-medical.git
git fetch origin
git checkout -b fix/ehpwas-binding origin/main
# copy is already the working tree; or:
# git add -A && git commit -m "fix: bind ZKP to HE ciphertext; drop trusted is_within_bound"
# git push -u origin fix/ehpwas-binding
```

Or from an existing clone, copy:

- `crypto/zkp_norm.py`
- `experiments/run_experiment.py`
- `crypto/homomorphic.py` (comment only)
- `SECURITY.md` (this file)

then commit/push.
