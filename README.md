# S301_S5-HVS1_Bayes

Reproduction code for the statistical analysis of

> **Is S301 the Captured Companion of the Hypervelocity Star S5-HVS1?**
> Andrea Caputo and Giovanni Maria Tomaselli

Stellar binary disruptions by Sgr A* (the Hills mechanism) leave two fossils: a
hypervelocity star and a star tightly bound to the black hole. This repository
implements the forward model that asks whether the newly discovered S-star
**S301** is the captured former companion of the hypervelocity star
**S5-HVS1**, and computes every number of the paper's statistical analysis —
in particular the per-star and catalog-level Bayes factors.

## Results

The fiducial model has density slope γ = 1.1 and a thermal null eccentricity
distribution. `python scripts/robustness.py` reproduces the paper's table of
fiducial results and robustness checks (one ingredient changed at a time):

|            | fiducial        | γ = 1.75        | α = 1.7         | with mass       |
|------------|-----------------|-----------------|-----------------|-----------------|
| B_301      | 90              | 42              | 68              | 250             |
| B_14       | 2.4             | 3.2             | 1.8             | 1.8             |
| B_29       | 0.58            | 0.90            | 0.44            | 1.7             |
| B_175      | 0.59            | 1.1             | 0.44            | 0.0028          |
| B_cat      | 1 + 4.23 p_det  | 1 + 1.65 p_det  | 1 + 2.92 p_det  | 1 + 13.0 p_det  |

The script also prints a secondary α = 2.6 column (see "Robustness checks"
below). `python scripts/run_analysis.py` prints the full per-star output of
the orbit-only analysis for the fiducial γ = 1.1 and for γ = 1.75 (all 18
individual Bayes factors, survival probabilities, and the Bayes factor of
S301's mirrored orbital solution). The calculation is fully deterministic —
repeated runs give identical numbers — and takes about ten seconds.
`python scripts/paper_numbers.py` reproduces the auxiliary numbers quoted in
the text (line coefficient, predicted and photometric masses, Δj, the α fit,
the inputs of the mass check, etc.), each printed next to the quoted value.

| in the paper | reproduced by |
|---|---|
| table of Bayes factors (fiducial and the three robustness checks) | `scripts/robustness.py` |
| all 18 individual B_i, survival probabilities, S301's mirrored solution | `scripts/run_analysis.py` |
| details and sensitivity of the mass check | `scripts/robustness_mass.py` |
| outlook: S301 with a spectroscopic mass measurement | `scripts/robustness_mass.py` (FORECAST block) |
| numbers quoted in the text (Eq. (1), masses, Δj, N, α, …) | `scripts/paper_numbers.py` |

## The model

**Pair hypothesis.** The disrupted binary is drawn from a Kroupa (2001) IMF
with mass ratio q^-0.3 on [0.1, 1] plus a 10% twin excess uniform on [0.9, 1]
(Moe & Di Stefano 2017). Either star is the ejected one with equal prior
probability 1/2, weighted by the Gaussian likelihood of S5-HVS1's measured
mass, 2.35 ± 0.06 M☉; the captured companion is the other member, restricted
to m_b ∈ [1.1, 7] M☉ (GRAVITY visibility and main-sequence lifetime).
Energy conservation then fixes the companion's semi-major axis on the line
a = G M_• m_b / (m_e v_∞²) ≈ 502 AU (m_b/M☉). The capture eccentricity
follows 1 − e₀ = κ (m_bin/M_•)^(1/3) with κ lognormal (median 1.5,
σ_lnκ = 0.45), calibrated on the binary-disruption simulations of Generozov
et al. (2025). Over S5-HVS1's flight time t_f = 4.8 Myr, the angular momentum
j = √(1−e²) diffuses with constant coefficient D_jj = j_301²/t_j(a), where
t_j(a) = 3×10⁷ yr (a/687 AU)^(γ−3/2), including the drift term D_jj/(2j)
required for the thermal distribution p(j) ∝ j to be stationary:

    dj = [D_jj/(2j)] dt + √D_jj dW ,

with a reflecting boundary at j = 1 and an absorbing boundary at pericenter
r_p = 3 AU (cumulative tidal heating). The likelihood f_pair(a, e) is the
resulting density of survivors.

**Null hypothesis.** Thermal eccentricities, p(e) = 2e, and semi-major axes
following the density cusp ρ ∝ r^-γ (i.e. p(a) ∝ a^(2−γ)), normalized over
the same semi-major-axis interval. The fiducial slope is γ = 1.1
(extrapolated diffuse light); γ = 1.75 (Bahcall–Wolf) is a robustness check
(see below).

**Bayes factors.** B_i = f_pair(a_i, e_i)/f_0(a_i, e_i) for each of the
N = 18 stars with a inside the support of f_pair ([552, 3511] AU, quoted
rounded as [550, 3500] AU in the paper), and the look-elsewhere-corrected
catalog-level factor B_cat = (1 − p_det) + (p_det/N) Σᵢ B_i, where p_det is
the probability that the companion survived and is detected in the catalog.

## Robustness checks

**Density slope.** γ = 1.75 (Bahcall–Wolf) instead of γ = 1.1, in both the
null a-distribution and the scaling of t_j.

**Null eccentricity distribution.** p(e) = (1+α) e^α instead of the thermal
α = 1. The tabulated α = 1.7 is the maximum-likelihood fit to the
eccentricities of the 17 catalog stars other than S301,
α̂ = −n/Σᵢ ln e_i − 1 = 1.73 ± 0.66 (Fisher error (1+α̂)/√n;
`likelihoods.fit_alpha`), i.e. α = 1.7 ± 0.7. `python scripts/robustness.py`
also prints α = 2.6, the fit of Gillessen et al. 2009 to the early-type stars
outside the clockwise disk (2.6 ± 0.9). Both are in
`ALPHAS_ROBUSTNESS = (1.7, 2.6)` in `constants.py`; other values via
`python scripts/robustness.py --alpha 2 3 4`. Only the null depends on α, so
B_i(α) = B_i(1) · 2 e_i^(1−α)/(1+α).

**Mass.** `python scripts/robustness_mass.py` adds each star's observed K
magnitude K_i as a further observable:

    B_i^mass = B_i · (Z/Z_s) · L_pair(K_i | a_i) / p_0(K_i | det) .

Both hypotheses share the mass–magnitude relation
K(m) = M_Ks(m) + 5 log10(R0/10 pc) + A_K, with R0 = 8178 pc, A_K = 2.55 and
M_Ks(m) interpolated linearly in log10 m through Mamajek's main-sequence dwarf
table (v2022.04.16); a Gaussian scatter σ_K = 0.4 mag of K about K(m) (0.3 mag
photometric error of S301, ~0.15 mag extinction, ~0.2 mag scatter of the
relation, in quadrature); and the detection limit K ≤ K_lim = 20 (GRAVITY's
limiting magnitude). Under the pair hypothesis the true mass lies on the line
within its ~3% width, m ~ N(a/C, 0.03 a/C), so

    L_pair(K | a) = ∫ dm N(m; a/C, 0.03 a/C) N(K; K(m), σ_K) ,

and the fiducial companion-mass prior (m_b ∈ [1.1, 7] M☉) is further
conditioned on detection: the normalization Z = ∫ da u(a) P_surv(a) of
f_pair becomes Z_s = ∫ da u(a) P_surv(a) Φ((K_lim − K(a/C))/σ_K), with Φ the
standard normal cdf (the line width is neglected inside Φ, a 5×10⁻⁴ effect on
Z/Z_s), and Z/Z_s = 1.025. Under the null, masses follow the Kroupa IMF
ξ ∝ m^-2.3 on [0.5, 20] M☉, normalized over the detected stars:

    p_0(K | det) = ∫ dm ξ(m) N(K; K(m), σ_K) / ∫ dm ξ(m) Φ((K_lim − K(m))/σ_K) .

All 18 stars use the same recipe (K magnitudes from Gillessen et al. 2017,
Table 3; S301: K = 19.3), including the late-type giants S17, S21 and S38, for
which the main-sequence relation is only formal. The script prints the
per-star factors with each star's photometric mass for reference (S301:
1.38 M☉), and a sensitivity block varying σ_K (0.3–0.5 mag), K_lim (19.7–20.5),
A_K (2.4–2.7), the line width (0–7%), the pair relation (a MIST isochrone
coeval with S5-HVS1, below), the ends of the null IMF (upper end 12 M☉,
which changes B_301 and the B_cat coefficient by less than 3%; lower end
0.3 M☉) and a brighter limit K_lim = 18 for the null of the 17
adaptive-optics-selected stars: B_301 stays between 172 and 316 and the
B_cat coefficient between 8.7 and 16.6. The isochrone is MIST v1.2 with
v/v_crit = 0.4, [Fe/H] = +0.25 and log10(age/yr) = 7.7, in 2MASS Ks: the grid
point closest to S5-HVS1's log10 age 7.72 and [Fe/H] = 0.29 ± 0.08 (Koposov
et al. 2020).

**Forecast: a spectroscopic mass.** The same machinery gives the Bayes
factors that a future spectroscopic mass measurement of S301 would produce
(`mass.spectroscopic_forecast`). A measured mass m_spec, Gaussian about the
true mass with width σ_m, is one more observable next to K: both integrals
above acquire the factor N(m_spec; m, σ_m), and B_301^mass is multiplied by
the ratio of the conditional densities of m_spec given K under the two
hypotheses,

    [L_pair(K, m_spec | a) / L_pair(K | a)] / [p_0(K, m_spec | det) / p_0(K | det)] ,

while the other stars keep their B_i^mass. A measurement with 5% precision
lying on the energy line (m_spec = a/C = 1.37 M☉, σ_m = 0.05 m_spec) gives
B_301 = 650 and B_cat = 1 + 35 p_det; because of the 3% width of the line,
even a perfect measurement cannot exceed B_cat ≈ 1 + 67 p_det. A measurement
away from the line removes the evidence instead: B_cat < 1 for
m_spec ≤ 1.1 M☉ or m_spec ≥ 1.7 M☉. The FORECAST block of
`scripts/robustness_mass.py` prints these numbers for precisions of 10%, 5%
and 3% and for a range of m_spec.

## Data

Orbital elements are from Gillessen et al. 2017 (ApJ 837, 30, Table 3) for
most stars, overridden by the GRAVITY Collaboration 2024 four-star fit
(A&A 692, A242, Table B.1) for S2, S29, S38, S55, plus S301 from GRAVITY
Collaboration 2026 (arXiv:2607.12664). The mass check uses the K magnitudes
of Gillessen et al. 2017 (Table 3) and GRAVITY Collaboration 2026 (S301),
Mamajek's main-sequence dwarf table (Pecaut & Mamajek 2013, v2022.04.16) and
one MIST v1.2 isochrone (Choi et al. 2016; Dotter 2016). See `data/README.md`
for details and distance conversions.

## Numerics and validation

The primary pipeline is deterministic: the companion-mass distribution is
computed by direct quadrature, and the j-diffusion by a Crank–Nicolson
Fokker–Planck solve in a flux form that preserves the thermal state to
machine precision (`src/s301_bayes/evolution.py`). A literal seeded
Monte Carlo realization of the forward model (brute-force binary sampling +
Euler–Maruyama random walk with Brownian-bridge boundary crossing) is
included and agrees with the deterministic numbers at the sub-percent level:

    python scripts/mc_crosscheck.py

The test suite (`pytest`, ~1 min) guards the physics choices that matter:
thermal stationarity of the drift term, the equal-ejection-prior treatment of
the two binary members (exchangeability once the Gaussian mass likelihood is
divided out), the line coefficient, the catalog selection, and
Fokker–Planck ↔ Monte Carlo agreement; for the mass check and the forecast,
it validates every magnitude-space integral against adaptive quadrature and
the normalizations of all densities. It also pins the numbers quoted in the
paper as regression values. Doubling all grid resolutions changes the Bayes factors
at the 10⁻⁵ level.

## Install and run

Requires Python ≥ 3.9 with `numpy` (≥ 1.24, including 2.x) and `scipy`
(≥ 1.10); the tests need `pytest`. No installation is needed to run from a
clone:

    python scripts/run_analysis.py      # fiducial + γ=1.75, all stars (~10 s)
    python scripts/robustness.py        # the paper's robustness table (~10 s)
    python scripts/robustness_mass.py   # mass check, sensitivity, forecast (~5 s)
    python scripts/paper_numbers.py     # auxiliary numbers in the text (~5 s)
    python -m pytest                    # test suite                   (~1 min)
    python scripts/mc_crosscheck.py     # high-precision MC validation (~30 min)

or install the package with `pip install -e .` (`pip install -e ".[test]"`
to include `pytest`) and use it as a library:

    from s301_bayes import compute_catalog, compute_catalog_mass, spectroscopic_forecast

    orbit = compute_catalog(1.1)             # fiducial; orbit["S301"].bayes, orbit.bcat_coeff
    mass = compute_catalog_mass(1.1)         # with K magnitudes
    outlook = spectroscopic_forecast(mass)   # S301 with a 5% spectroscopic mass

All model parameters are defined, with their sources, in
`src/s301_bayes/constants.py`.

## Citation

If you use this code, please cite the paper above.

## License

MIT — see `LICENSE`.
