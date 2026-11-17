"""Power-law null eccentricity distribution p(e) = (1+alpha) e^alpha, the
robustness check of the thermal null (alpha = 1, the fiducial choice). Only
f_0 depends on alpha; the pair model is unchanged."""

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.optimize import minimize_scalar

from s301_bayes import constants as c
from s301_bayes.bayes import compute_catalog
from s301_bayes.catalog import load_catalog, load_s301
from s301_bayes.likelihoods import (
    PairLikelihood,
    fit_alpha,
    null_e_pdf,
    null_likelihood,
)


def _thermal_null(a_au, e, gamma):
    """The original thermal null, f_0 = cusp(a) * 2e, verbatim."""
    a_lo, a_hi = c.A_SUPPORT
    a_au = np.asarray(a_au, dtype=float)
    g = 3.0 - gamma
    norm = g / (a_hi**g - a_lo**g)
    out = norm * a_au ** (2.0 - gamma) * 2.0 * np.asarray(e, dtype=float)
    return np.where((a_au >= a_lo) & (a_au <= a_hi), out, 0.0)


@pytest.fixture(scope="module")
def pair_fiducial():
    return PairLikelihood(c.GAMMA_FIDUCIAL)


@pytest.mark.parametrize("gamma", c.GAMMAS)
def test_alpha1_is_thermal_bit_for_bit(gamma):
    assert c.ALPHA_THERMAL == 1.0
    rng = np.random.default_rng(7)
    a = rng.uniform(400.0, 4000.0, 2000)  # also outside the support
    e = rng.uniform(0.0, 1.0, 2000)
    old = _thermal_null(a, e, gamma)
    assert np.array_equal(null_likelihood(a, e, gamma), old)
    assert np.array_equal(null_likelihood(a, e, gamma, alpha=1.0), old)
    assert null_likelihood(687.0, 0.9832, gamma) == float(_thermal_null(687.0, 0.9832, gamma))


@pytest.mark.parametrize("alpha", sorted({0.0, 0.5, 1.0, 2.0, 3.5, *c.ALPHAS_ROBUSTNESS}))
def test_e_pdf_normalized(alpha):
    val, _ = quad(null_e_pdf, 0.0, 1.0, args=(alpha,))
    assert abs(val - 1.0) < 1e-10


@pytest.mark.parametrize("alpha", [0.0, 3.5])
@pytest.mark.parametrize("gamma", c.GAMMAS)
def test_null_density_normalized(gamma, alpha):
    val, _ = dblquad(
        lambda e, a: null_likelihood(a, e, gamma, alpha), *c.A_SUPPORT, 0.0, 1.0
    )
    assert abs(val - 1.0) < 1e-8


def test_non_normalizable_alpha_rejected():
    with pytest.raises(ValueError):
        null_e_pdf(0.5, -1.0)


@pytest.mark.parametrize("alpha", [0.0, *c.ALPHAS_ROBUSTNESS])
def test_bayes_ratio_per_star(pair_fiducial, alpha):
    """B_i(alpha) / B_i(1) = 2 e_i^(1-alpha) / (1+alpha), f_pair untouched."""
    g = c.GAMMA_FIDUCIAL
    ref = compute_catalog(g, pair_fiducial)
    alt = compute_catalog(g, pair_fiducial, alpha=alpha)
    assert (ref.alpha, alt.alpha) == (1.0, alpha)
    assert alt.n_stars == ref.n_stars == 18
    for r0, r1 in zip(ref.results, alt.results):
        assert r1.star == r0.star
        assert r1.f_pair == r0.f_pair
        expected = 2.0 * r0.star.e ** (1.0 - alpha) / (1.0 + alpha)
        assert np.isclose(r1.bayes / r0.bayes, expected, rtol=1e-12, atol=0.0)
    e_m = load_s301("mirror").e
    expected = 2.0 * e_m ** (1.0 - alpha) / (1.0 + alpha)
    assert np.isclose(alt.mirror_b301 / ref.mirror_b301, expected, rtol=1e-12, atol=0.0)


def test_fit_alpha_catalog():
    """alpha = 1.7 +- 0.7 is the maximum-likelihood exponent of the 17 catalog
    stars other than S301 with its Fisher error (1.73 +- 0.66), rounded."""
    e = [s.e for s in load_catalog() if s.name != "S301"]
    assert len(e) == 17
    alpha, sigma = fit_alpha(e)
    assert (round(alpha, 2), round(sigma, 2)) == (1.73, 0.66)
    assert round(alpha, 1) == c.ALPHAS_ROBUSTNESS[0] == 1.7 and round(sigma, 1) == 0.7
    # it maximizes the log-likelihood sum ln p(e_i) ...
    nll = lambda a: -np.sum(np.log(null_e_pdf(e, a)))
    best = minimize_scalar(nll, bounds=(0.0, 5.0), method="bounded", options={"xatol": 1e-10})
    assert best.x == pytest.approx(alpha, abs=1e-6)
    # ... and sigma^-2 is its curvature there (the Fisher information)
    h = 1e-3
    curv = (nll(alpha + h) - 2.0 * nll(alpha) + nll(alpha - h)) / h**2
    assert sigma == pytest.approx(curv**-0.5, rel=1e-6)
    assert c.ALPHAS_ROBUSTNESS[1] == 2.6  # Gillessen et al. 2009


def test_fit_alpha_recovers_exponent():
    rng = np.random.default_rng(11)
    e = rng.random(200_000) ** (1.0 / 3.6)   # p(e) = 3.6 e^2.6
    alpha, sigma = fit_alpha(e)
    assert alpha == pytest.approx(2.6, abs=0.03)
    assert sigma == pytest.approx(3.6 / np.sqrt(e.size), rel=0.01)
    for bad in ([], [0.5, 1.0], [0.0, 0.5], [[0.5, 0.6]]):
        with pytest.raises(ValueError):
            fit_alpha(bad)


def test_alpha_1p7_numbers(pair_fiducial):
    """The tabulated alpha = 1.7 column (regression values)."""
    res = compute_catalog(c.GAMMA_FIDUCIAL, pair_fiducial, alpha=1.7)
    got = [res[n].bayes for n in ("S301", "S14", "S29", "S175")] + [res.bcat_coeff]
    assert [float(f"{x:.3g}") for x in got] == [67.6, 1.77, 0.438, 0.440, 2.92]


def test_fiducial_headline_numbers_unchanged(pair_fiducial):
    assert c.GAMMA_FIDUCIAL == 1.1
    res = compute_catalog(c.GAMMA_FIDUCIAL, pair_fiducial)
    assert res.alpha == c.ALPHA_THERMAL
    assert f"{res['S301'].bayes:.3g}" == "90.2"
    assert f"{res.bcat_coeff:.3g}" == "4.23"


def test_pair_gamma_mismatch_rejected(pair_fiducial):
    with pytest.raises(ValueError):
        compute_catalog(1.75, pair_fiducial)
