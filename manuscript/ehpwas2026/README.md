# Manuscript sources (IEEE ≤6 pages)

Canonical PDF: `main.pdf` (built from `main.tex`).

| Item | Value |
|------|-------|
| Class | `IEEEtran` conference |
| Page limit | ≤ 6 IEEE pages |
| PDF eXpress ID | `61911X` |
| Figures | `figures/*.png` |
| Artifact | https://github.com/edlansiaux/pq-zkfl-medical (`main`) |

## Build

```bash
# From this directory, with Times fonts available (see author tree) or TeX Live:
tectonic main.tex
# or: pdflatex main.tex && pdflatex main.tex
```

Numeric claims should match JSON under `../../results/`.
