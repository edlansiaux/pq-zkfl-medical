# ZKFL-PQ — Camera-ready (eHPWAS 2026 / WiMob)

`main.pdf` — **6 IEEE pages**, EDAS margins OK, no Type-3 fonts.

| | |
|--|--|
| Artifact branch | `fix/ehpwas-binding` |
| PR | https://github.com/edlansiaux/pq-zkfl-medical/pull/1 |
| PDF eXpress ID | **61911X** |
| Deadline | **1 September 2026** |

## Reviewer A–D mapping (in the PDF)

| Theme | Camera-ready response |
|--------|----------------------|
| A weak eval | 5-seed baselines + sign-flip + UCI Breast Cancer + ablations |
| B/C crypto | Bound language, Unruh NIZK, BFV table, threshold, security analysis |
| D code ≠ claims | Artifact aligned (binding, full-vector HE, no trusted Boolean) |
| Beskar | Related-work table + non-replacement positioning |
| WiMob | Payload tables (KB/MB) + scaling |

## PDF eXpress / EDAS

1. Proofread `main.pdf`.
2. https://ieee-pdf-express.org — conference ID **61911X**.
3. Download PDF eXpress–approved file.
4. IEEE copyright via EDAS.
5. Upload approved PDF + registration code.
6. At least one **regular (non-student)** registration.

Page limit: **6** IEEE pages (up to 2 extra at 100€ each).

## Build locally

From this folder (with Times fonts / Tectonic as in the author tree):

```bash
# example with tectonic
tectonic -X compile main.tex
```

Figures are PNG (avoid Type-3 DejaVu from matplotlib PDFs).
