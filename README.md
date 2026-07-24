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

`python scripts/run_analysis.py` reproduces the paper table:

|            | γ = 1.1          | γ = 1.75         |
|------------|------------------|------------------|
| B_301      | 90               | 42               |
| B_14       | 2.4              | 3.2              |
| B_29       | 0.58             | 0.90             |
| B_175      | 0.59             | 1.1              |
| B_cat      | 1 + 4.23 p_det   | 1 + 1.65 p_det   |

(plus all 18 individual Bayes factors, survival probabilities, and the Bayes
factor of S301's mirrored orbital solution). The calculation is fully
deterministic — repeated runs give identical numbers — and takes about ten
seconds. `python scripts/paper_numbers.py` reproduces the auxiliary numbers
quoted in the text (line coefficient, predicted masses, Δj, etc.).

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
the same semi-major-axis interval, for γ = 1.1 (extrapolated diffuse light)
and γ = 1.75 (Bahcall–Wolf).

**Bayes factors.** B_i = f_pair(a_i, e_i)/f_0(a_i, e_i) for each of the
N = 18 stars with a inside the support of f_pair ([552, 3511] AU, quoted
rounded as [550, 3500] AU in the paper), and the look-elsewhere-corrected
catalog-level factor B_cat = (1 − p_det) + (p_det/N) Σᵢ B_i, where p_det is
the probability that the companion survived and is detected in the catalog.

## Data

Orbital elements are from Gillessen et al. 2017 (ApJ 837, 30, Table 3) for
most stars, overridden by the GRAVITY Collaboration 2024 four-star fit
(A&A 692, A242, Table B.1) for S2, S29, S38, S55, plus S301 from GRAVITY
Collaboration 2026 (arXiv:2607.12664). See `data/README.md` for details and
distance conversions.

## Numerics and validation

The primary pipeline is deterministic: the companion-mass distribution is
computed by direct quadrature, and the j-diffusion by a Crank–Nicolson
Fokker–Planck solve in a flux form that preserves the thermal state to
machine precision (`src/s301_bayes/evolution.py`). A literal seeded
Monte Carlo realization of the forward model (brute-force binary sampling +
Euler–Maruyama random walk with Brownian-bridge boundary crossing) is
included and agrees with the deterministic numbers at the sub-percent level:

    python scripts/mc_crosscheck.py

The test suite (`pytest`, ~20 s) guards the physics choices that matter:
thermal stationarity of the drift term, the equal-ejection-prior treatment of
the two binary members (exchangeability once the Gaussian mass likelihood is
divided out), the line coefficient, the catalog selection, and
Fokker–Planck ↔ Monte Carlo agreement. Doubling all grid resolutions changes
the Bayes factors at the 10⁻⁵ level.

## Install and run

Requires Python ≥ 3.9 with `numpy` and `scipy`. No installation is needed to
run from a clone:

    python scripts/run_analysis.py      # the Bayes-factor table  (~10 s)
    python scripts/paper_numbers.py     # auxiliary numbers in the text
    python -m pytest                    # test suite              (~15 s)
    python scripts/mc_crosscheck.py     # high-precision MC validation (~30 min)

or install the package with `pip install -e .` and use
`from s301_bayes import compute_all`.

## Citation

If you use this code, please cite the paper above.

## License

MIT — see `LICENSE`.
