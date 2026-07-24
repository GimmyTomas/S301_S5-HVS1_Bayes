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
A_301 = 687.0              # AU
E_301 = 0.9832             # primary orbital solution
J_301 = float(np.sqrt(1.0 - E_301**2))   # ~ 0.1825

# ----------------------------------------------------------------------------
# Angular-momentum diffusion:  t_j(a) = T_J_301 * (a/A_301)^(gamma - 3/2),
# D_jj = J_301^2 / t_j(a), independent of j (non-resonant two-body relaxation).
# ----------------------------------------------------------------------------
T_J_301 = 3.0e7            # yr, j-diffusion time at a = A_301
GAMMAS = (1.1, 1.75)       # density-profile slopes rho ~ r^-gamma considered

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
IMF_ALPHA = 2.3            # Kroupa (2001) IMF slope for m > 0.5 Msun; only this
                           # branch is ever sampled (all relevant masses > 1.1)
Q_ALPHA = -0.3             # p(q) ~ q^Q_ALPHA on [Q_MIN, 1] (Moe & Di Stefano 2017)
Q_MIN = 0.1
F_TWIN = 0.10              # twin-excess fraction, uniform on [Q_TWIN_MIN, 1]
Q_TWIN_MIN = 0.9

# ----------------------------------------------------------------------------
# Catalogs: angular-to-physical conversion, a[AU] = a[arcsec] * R0[pc]
# ----------------------------------------------------------------------------
R0_GILLESSEN17_PC = 8320.0   # Gillessen et al. 2017 best fit (R0 = 8.32 kpc)
R0_GRAVITY24_PC = 8275.9     # GRAVITY Collaboration 2024, Table B.1
