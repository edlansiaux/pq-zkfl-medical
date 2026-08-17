# Audit notes — https://github.com/edlansiaux/pq-zkfl-medical

## Closed (former “out of scope”)

| Item | Implementation |
|------|----------------|
| Threshold BFV | `ThresholdBFV` + `GradientHEManager(use_threshold=True)` — `(2,3)` Shamir, no monolithic server `sk` |
| Real medical data | `load_medical_dataset("breast_cancer")` (UCI via sklearn); optional MedMNIST |
| QROM-oriented NIZK | `crypto/qrom_nizk.py` Unruh-style parallel binary sessions |
| Full-vector HE | All chunks encrypted; demo MLP on Breast Cancer (`run_target_protocol.py`) |
| ~128-bit HE class | `HE_CLAIMED_SECURITY_BITS=128` target class; demo `HE_N=512` (prod tables often `n=4096`) — **not** SEAL-certified |

## Demo result (`results/target_protocol_results.json`)

- UCI Breast Cancer, 5 clients, 3 rounds, Unruh `r=16`, threshold `(2,3)`
- Final acc ≈ **0.94**; malicious client rejected on attack round
- Full-vector HE + threshold decrypt + Unruh all ON

## Still not “certificate grade”

- Lattice-estimator / SEAL backend at production `n=4096`
- Distributed decrypt **without** reconstructing `sk`
- Machine-checked QROM proof
