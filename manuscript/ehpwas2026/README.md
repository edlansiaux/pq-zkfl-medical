# Manuscript sources (IEEE ≤6 pages)

Canonical files in **this folder**: `main.tex`, `main.pdf`, `figures/*.png`.

| Item | Value |
|------|-------|
| Class | `IEEEtran` conference |
| Page limit | ≤ 6 IEEE pages (densely filled) |
| PDF eXpress ID | `61911X` |
| Artifact | https://github.com/edlansiaux/pq-zkfl-medical (`main`) |
| Numbers | Must match `../../results/*.json` |

## Build

```bash
tectonic main.tex
# or: pdflatex main.tex && pdflatex main.tex
```

Do not use root `../../figures/` for the camera-ready PDF; TeX includes only `./figures/`.
