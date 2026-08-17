# ZKFL-PQ — Camera-ready (eHPWAS 2026 / WiMob)

`main.pdf` — **6 IEEE pages**, marges OK, pas de Type-3.  
Artifact: `fix/ehpwas-binding` @ `7a0f79b`  
PR: https://github.com/edlansiaux/pq-zkfl-medical/compare/main...fix/ehpwas-binding?expand=1

## Réponse reviewers A–D (dans le PDF)

| Review | Traitement camera-ready |
|--------|-------------------------|
| A éval faible | Baselines 5 seeds + sign-flip + UCI Breast Cancer + ablations |
| B/C crypto | $\mathcal{L}^{bind}$, Unruh, BFV table, threshold, analyse sécurité |
| D code≠claims | Artifact aligné (binding, full-vector HE, pas de Boolean de confiance) |
| Beskar | Table related work + positionnement non-remplacement |
| WiMob | Payloads KB/MB + scaling $O(N\lceil d/n\rceil)$ |

## PDF eXpress

Conference ID **61911X** — deadline **1 Sep 2026**.

## IEEE PDF eXpress (final manuscript)

1. Proofread `main.pdf`.
2. Create/login at https://ieee-pdf-express.org/account/login  
   - **Conference ID (eHPWAS):** `61911X` (as in acceptance mail).
3. Convert/check the PDF with PDF eXpress; download the approved file.
4. Complete **IEEE Copyright Form** via EDAS → IEEE Copyright Submission.
5. Upload the PDF eXpress–approved PDF to EDAS using your **registration code**.
6. Deadline (camera-ready + author registration): **1 September 2026**.
7. At least one **regular (non-student)** registration is required.

Page limit: **6 IEEE pages** (up to 2 extra pages at 100€ each).

## Build locally

```powershell
cd C:\Users\edlsx\zkfl-ehwasp2026
.\bin\tectonic.exe -X compile main.tex
```

Figures are PNG (no Type-3 DejaVu fonts). Margins match prior EDAS checks (top ≥0.85 in, bottom ≥1.08 in).
