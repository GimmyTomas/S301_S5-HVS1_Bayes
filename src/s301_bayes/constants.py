"""All physical constants and model parameters of the analysis.

Every number entering the statistical analysis of the paper is defined here,
with its source. Nothing in the rest of the package hard-codes a parameter.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Unit conversions (IAU nominal values)
# ----------------------------------------------------------------------------
GM_SUN = 1.32712440018e20  # m^3 s^-2, solar mass parameter
AU = 1.495978707e11        # m
YEAR = 365.25 * 86400.0    # s (Julian year)

# ----------------------------------------------------------------------------
# Sgr A* and S5-HVS1
# ----------------------------------------------------------------------------
M_BH = 4.30e6              # Msun, Sgr A* mass (GRAVITY Collaboration 2022)
M_EJ = 2.35                # Msun, S5-HVS1 mass (Koposov et al. 2020)
M_EJ_SIGMA = 0.06          # Msun, Gaussian measurement likelihood width
# integration/sampling truncation of the Gaussian (weight < 1e-14 outside)
M_EJ_RANGE = (M_EJ - 8.0 * M_EJ_SIGMA, M_EJ + 8.0 * M_EJ_SIGMA)
V_INF = 1799.0e3           # m/s, S5-HVS1 asymptotic speed (Koposov et al. 2020)
T_FLIGHT = 4.8e6           # yr, flight time of S5-HVS1

# ----------------------------------------------------------------------------
# The mass-energy line, Eq. (1) of the paper:  a_b = LINE_COEFF_AU * (m_b/Msun)
# ----------------------------------------------------------------------------
LINE_COEFF_AU = GM_SUN * M_BH / (M_EJ * V_INF**2) / AU   # ~ 501.6 AU/Msun

# Companion-mass limits: m_b >= 1.1 Msun (GRAVITY visibility, m_K <~ 20) and
# m_b < 7 Msun (main-sequence lifetime longer than S5-HVS1's ~50 Myr age).
MB_MIN = 1.1               # Msun
MB_MAX = 7.0               # Msun

# Semi-major axis support of the pair hypothesis (the paper quotes the rounded
# interval [550, 3500] AU).
A_SUPPORT = (MB_MIN * LINE_COEFF_AU, MB_MAX * LINE_COEFF_AU)

# ----------------------------------------------------------------------------
# S301 (GRAVITY Collaboration 2026) -- reference orbit for the diffusion time
# ----------------------------------------------------------------------------
A_301 = 687.0              # AU (83.0 mas at R0 = 8275.9 pc, see data/s301.csv)
E_301 = 0.9832             # primary orbital solution
J_301 = float(np.sqrt(1.0 - E_301**2))   # ~ 0.1825

# ----------------------------------------------------------------------------
# Angular-momentum diffusion:  t_j(a) = T_J_301 * (a/A_301)^(gamma - 3/2),
# D_jj = J_301^2 / t_j(a), independent of j (non-resonant two-body relaxation).
# ----------------------------------------------------------------------------
T_J_301 = 3.0e7            # yr, j-diffusion time at a = A_301
GAMMAS = (1.1, 1.75)       # density-profile slopes rho ~ r^-gamma considered
GAMMA_FIDUCIAL = 1.1       # fiducial slope; the others in GAMMAS are robustness checks

# ----------------------------------------------------------------------------
# Null-hypothesis eccentricity distribution  p(e) = (1+alpha) e^alpha  on [0, 1].
# The fiducial null is thermal (alpha = 1, p(e) = 2e). Robustness checks:
#  * alpha = 1.7 (the tabulated value): the maximum-likelihood exponent of the
#    17 catalog stars other than S301, alpha = -n / sum(ln e_i) - 1 = 1.73,
#    with Fisher error (1+alpha) / sqrt(n) = 0.66 (likelihoods.fit_alpha),
#    i.e. 1.7 +- 0.7;
#  * alpha = 2.6, the fit of Gillessen et al. 2009 (ApJ 692, 1075) to the
#    early-type stars outside the clockwise disk (2.6 +- 0.9), kept as a
#    secondary value.
# Any alpha > -1 is allowed (scripts/robustness.py --alpha).
# ----------------------------------------------------------------------------
ALPHA_THERMAL = 1.0
ALPHAS_ROBUSTNESS = (1.7, 2.6)

# ----------------------------------------------------------------------------
# Initial (capture) eccentricity:  1 - e0 = kappa * (m_bin/M_BH)^(1/3),
# kappa lognormal, calibrated on the binary-disruption simulations of
# Generozov et al. 2025.
# ----------------------------------------------------------------------------
KAPPA_MEDIAN = 1.5
KAPPA_SIGMA_LN = 0.45

# ----------------------------------------------------------------------------
# Absorbing boundary: stars with pericenter below this distance are destroyed
# by cumulative tidal heating within the flight time (cf. Lu et al. 2021).
# ----------------------------------------------------------------------------
R_ABSORB_AU = 3.0          # AU (pericenter)

# ----------------------------------------------------------------------------
# Binary population model
# ----------------------------------------------------------------------------
IMF_ALPHA = 2.3            # Kroupa (2001) IMF slope for m > 0.5 Msun, the only
                           # branch needed: binary members are > 1.1 Msun, and the
                           # null of the mass check lives on M_NULL_RANGE
Q_ALPHA = -0.3             # p(q) ~ q^Q_ALPHA on [Q_MIN, 1] (Moe & Di Stefano 2017)
Q_MIN = 0.1
F_TWIN = 0.10              # twin-excess fraction, uniform on [Q_TWIN_MIN, 1]
Q_TWIN_MIN = 0.9

# ----------------------------------------------------------------------------
# Catalogs: angular-to-physical conversion, a[AU] = a[arcsec] * R0[pc]
# ----------------------------------------------------------------------------
R0_GILLESSEN17_PC = 8320.0   # Gillessen et al. 2017 best fit (R0 = 8.32 kpc)
R0_GRAVITY24_PC = 8275.9     # GRAVITY Collaboration 2024, Table B.1

# ----------------------------------------------------------------------------
# Mass robustness check, in K-magnitude space (s301_bayes.mass,
# scripts/robustness_mass.py). Mass-magnitude relation, the same for both
# hypotheses: K(m) = M_Ks(m) + DM + A_K, DM = 5 log10(R0 / 10 pc), with M_Ks(m)
# from Mamajek's main-sequence dwarf table (Pecaut & Mamajek 2013; version
# 2022.04.16, data/mamajek_dwarf_MKs_mass.csv).
# ----------------------------------------------------------------------------
R0_PHOT_PC = 8178.0        # pc, GRAVITY Collaboration 2019, A&A 625, L10
                           # (8178 +- 13 stat +- 22 sys pc)
A_K_RANGE = (2.4, 2.7)     # mag, K-band extinction towards the S-stars, range
                           # quoted in the paper (A_Ks = 2.42, Fritz et al. 2011,
                           # ApJ 737, 73, is the value used by GRAVITY+2026)
A_K_GC = 0.5 * (A_K_RANGE[0] + A_K_RANGE[1])   # 2.55 mag, midpoint
SIGMA_K = 0.4              # mag, Gaussian scatter of the observed K about K(m),
                           # all stars: 0.3 mag photometric error of S301
                           # (GRAVITY+2026), ~0.15 mag extinction (half-width of
                           # A_K_RANGE), ~0.2 mag intrinsic scatter of the
                           # relation; 0.39 mag in quadrature
K_LIM = 20.0               # mag, detection limit K <= K_LIM under both
                           # hypotheses (GRAVITY's limiting magnitude)
LINE_SIGMA_FRAC = 0.03     # fractional width of the energy line, sigma_m = 0.03 m:
                           # the ~3% uncertainty of the line quoted in the paper
                           # (from S5-HVS1's mass, M_EJ_SIGMA / M_EJ = 2.6%)
M_NULL_RANGE = (0.5, 20.0)  # Msun, support of the null mass function, Kroupa
                            # xi(m) ~ m^-IMF_ALPHA. Lowering the upper end to
                            # 12 Msun changes B_301 and the B_cat coefficient by
                            # less than 3% (2.3% and 2.4%), extending the lower
                            # end to 0.3 Msun by less than 1e-12
SPEC_MASS_FRAC = 0.05      # fractional precision of the future spectroscopic mass
                           # measurement of S301 forecast in the paper's outlook
                           # (mass.spectroscopic_forecast)
