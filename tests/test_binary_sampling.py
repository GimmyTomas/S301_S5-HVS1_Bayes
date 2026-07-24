"""Companion-mass distribution: quadrature vs brute-force Monte Carlo, and the
equal-prior treatment of which star of the binary is the ejected one (the
fixed sampling bug: both ordered configurations must contribute, each with
prior weight 1/2, with the Gaussian mass likelihood as the only asymmetry)."""

import numpy as np

from s301_bayes import constants as c
from s301_bayes.binary_sampling import (
    imf,
    mass_weight,
    p_mb,
    p_me_given_mb,
    q_pdf,
    sample_companions,
)


def test_p_mb_normalized():
    grid = np.linspace(c.MB_MIN, c.MB_MAX, 4001)
    assert abs(np.trapz(p_mb(grid), grid) - 1.0) < 1e-6


def test_quadrature_matches_bruteforce_mc():
    rng = np.random.default_rng(2)
    n = 2_000_000
    mb, _ = sample_companions(n, rng)
    counts, edges = np.histogram(mb, bins=48, range=(c.MB_MIN, c.MB_MAX))
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    # bin-averaged model probability (the pdf is steep, so average over the bin)
    fine = np.linspace(c.MB_MIN, c.MB_MAX, 4801)
    pdf_fine = p_mb(fine)
    expected = np.array(
        [
            np.trapz(pdf_fine[(fine >= lo) & (fine <= hi)], fine[(fine >= lo) & (fine <= hi)])
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    ) * n
    z = (counts - expected) / np.sqrt(np.maximum(expected, 1.0))
    assert np.max(np.abs(z)) < 5.0
    assert np.mean(np.abs(z)) < 1.5
    del centers, width


def test_exchangeability_after_removing_mass_weight():
    """With the Gaussian mass likelihood divided out, (m_b, m_ej) must be
    exchangeable (equal prior for either star to be ejected). The old code
    broke this symmetry for nearly-equal-mass binaries."""
    rng = np.random.default_rng(3)
    mb, me = sample_companions(2_000_000, rng)
    lo, hi = 2.25, 2.45  # band where both masses can play both roles
    sel = (mb >= lo) & (mb <= hi) & (me >= lo) & (me <= hi)
    assert sel.sum() > 20_000
    w = 1.0 / mass_weight(me[sel])  # undo the ejected-mass likelihood
    mean_b = np.average(mb[sel], weights=w)
    mean_e = np.average(me[sel], weights=w)
    ess = w.sum() ** 2 / (w**2).sum()
    sigma = np.sqrt(
        (np.average((mb[sel] - mean_b) ** 2, weights=w)
         + np.average((me[sel] - mean_e) ** 2, weights=w)) / ess
    )
    assert abs(mean_b - mean_e) < 5.0 * sigma


def test_pair_density_symmetry():
    """The joint (companion, ejected) density is symmetric up to the Gaussian
    mass likelihood of the ejected star: joint(x, y) * w(x) == joint(y, x) * w(y).
    Both ordered terms carry prior weight 1/2."""
    def joint(x, y):  # unnormalized: companion x, ejected y
        term_a = imf(y) * q_pdf(x / y) / y if x <= y else 0.0  # x is secondary
        term_b = imf(x) * q_pdf(y / x) / x if y <= x else 0.0  # x is primary
        return 0.5 * float(mass_weight(y)) * (float(term_a) + float(term_b))

    for x, y in [(2.25, 2.45), (2.3, 2.4), (2.2, 2.5)]:
        lhs = joint(x, y) * float(mass_weight(x))
        rhs = joint(y, x) * float(mass_weight(y))
        assert abs(lhs / rhs - 1.0) < 1e-12

    # p_me_given_mb implements the same ordered terms
    me = np.linspace(*c.M_EJ_RANGE, 601)
    cond = p_me_given_mb(me, 2.3)
    raw = np.array([joint(2.3, m) for m in me])
    raw /= np.trapz(raw, me)
    assert np.max(np.abs(cond - raw)) < 1e-10


def test_q_pdf_normalized_and_twin_excess():
    q = np.linspace(c.Q_MIN, 1.0, 200_001)
    # trapz across the twin-box discontinuity limits the accuracy to ~jump*h/2
    assert abs(np.trapz(q_pdf(q), q) - 1.0) < 1e-5
    # mass in [Q_TWIN_MIN, 1] = power-law share + the full twin excess
    pl_exp = 1.0 + c.Q_ALPHA
    pl_share = (1.0 - c.Q_TWIN_MIN**pl_exp) / (1.0 - c.Q_MIN**pl_exp)
    expected = (1.0 - c.F_TWIN) * pl_share + c.F_TWIN
    sel = q >= c.Q_TWIN_MIN
    assert abs(np.trapz(q_pdf(q)[sel], q[sel]) - expected) < 1e-4


def test_imf_slope():
    assert np.isclose(imf(2.0) / imf(4.0), 2.0**c.IMF_ALPHA)
