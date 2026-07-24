# Data provenance

All orbital elements used in the analysis, with their sources. The CSVs are
version-controlled so the paper numbers are reproducible offline; the
Gillessen table can be re-downloaded with `scripts/fetch_gillessen_table.py`.

| file | source | contents |
|---|---|---|
| `gillessen2017_table3.csv` | Gillessen et al. 2017, ApJ 837, 30, Table 3 (machine-readable version: VizieR catalog [J/ApJ/837/30](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/837/30)) | all 40 Galactic-center orbits (a in arcsec, e); S111 is unbound and dropped by the code |
| `gravity2024_tableB1.csv` | GRAVITY Collaboration 2024, A&A 692, A242 ([arXiv:2409.12261](https://arxiv.org/abs/2409.12261)), Table B.1 | updated orbits of S2, S29, S38, S55, overriding the Gillessen values |
| `s301.csv` | GRAVITY Collaboration 2026 ([arXiv:2607.12664](https://arxiv.org/abs/2607.12664)) | S301's two orbital solutions (a in AU) |

Angular semi-major axes are converted to AU with the distance fitted by the
respective paper: `a[AU] = a[arcsec] * R0[pc]`, with R0 = 8320 pc for
Gillessen et al. 2017 and R0 = 8275.9 pc for GRAVITY 2024 (see
`src/s301_bayes/constants.py` and `catalog.py`).
