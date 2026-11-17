# Data provenance

All orbital elements used in the analysis, and the photometric inputs and
mass–magnitude relations of the mass robustness check, with their sources.
The CSVs are version-controlled so the paper numbers are reproducible
offline; the Gillessen table can be re-downloaded with
`scripts/fetch_gillessen_table.py`.

| file | source | contents |
|---|---|---|
| `gillessen2017_table3.csv` | Gillessen et al. 2017, ApJ 837, 30, Table 3 (machine-readable version: VizieR catalog [J/ApJ/837/30](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/837/30)) | all 40 Galactic-center orbits (a in arcsec, e); S111 is unbound and dropped by the code |
| `gravity2024_tableB1.csv` | GRAVITY Collaboration 2024, A&A 692, A242 ([arXiv:2409.12261](https://arxiv.org/abs/2409.12261)), Table B.1 | updated orbits of S2, S29, S38, S55, overriding the Gillessen values |
| `s301.csv` | GRAVITY Collaboration 2026 ([arXiv:2607.12664](https://arxiv.org/abs/2607.12664)) | S301's two orbital solutions; the paper gives a = 83.0 ± 0.7 mas, converted here to a = 687 ± 6 AU with R0 = 8275.9 pc |
| `photometry.csv` | Gillessen et al. 2017, Table 3 (columns m_K and Sp); S301: GRAVITY Collaboration 2026 (m_K = 19.3 ± 0.3) | observed K magnitudes and spectral class (e/l/–) of the 39 bound stars and S301; used only by the mass robustness check |
| `mamajek_dwarf_MKs_mass.csv` | E. Mamajek, [EEM_dwarf_UBVIJHK_colors_Teff.txt](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt), version 2022.04.16 (Pecaut & Mamajek 2013, ApJS 208, 9) | columns SpT, Msun, M_Ks of the mean main-sequence dwarf table (all 118 rows; the 80 with both Msun and M_Ks, 0.075–20.2 M☉, define the mass–magnitude relation of the mass check) |
| `mist_v1.2_feh+0.25_logage7.7_MKs.csv` | MIST v1.2 isochrones ([waps.cfa.harvard.edu/MIST](https://waps.cfa.harvard.edu/MIST/); Choi et al. 2016, ApJ 823, 102; Dotter 2016, ApJS 222, 8), v/v_crit = 0.4, UBVRIplus photometry | initial mass, absolute 2MASS Ks magnitude and phase (pre-main sequence / main sequence) of the [Fe/H] = +0.25, log10(age/yr) = 7.7 isochrone (274 points, 0.52–6.92 M☉), coeval with S5-HVS1: the grid point closest to its log10 age 7.72 and [Fe/H] = 0.29 ± 0.08 (Koposov et al. 2020); pair-relation variant of the mass check |

Angular semi-major axes are converted to AU with the distance fitted by the
respective paper: `a[AU] = a[arcsec] * R0[pc]`, with R0 = 8320 pc for
Gillessen et al. 2017 and R0 = 8275.9 pc for GRAVITY 2024 (see
`src/s301_bayes/constants.py` and `catalog.py`).

The mass check (`src/s301_bayes/mass.py`) predicts the observed magnitude of
a star of mass m as K(m) = M_Ks(m) + DM + A_K, with DM = 5 log10(R0/10 pc),
R0 = 8178 pc (GRAVITY Collaboration 2019) and A_K = 2.55 (midpoint of
2.4–2.7). M_Ks is interpolated linearly in log10 m through the Mamajek rows
sorted by mass, or, for the pair hypothesis of the MIST variant, through the
isochrone (MIST v1.2, v/v_crit = 0.4, [Fe/H] = +0.25, log10(age/yr) = 7.7,
2MASS Ks; held constant beyond its last point, 6.915 M☉). The photometric
masses it displays invert the Mamajek relation (log10 M interpolated
linearly in M_Ks).
