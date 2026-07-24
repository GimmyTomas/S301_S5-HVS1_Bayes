"""The deterministic Fokker-Planck pipeline against the seeded Monte Carlo
realization of the same random walk (moderate sample size; the full-precision
comparison is scripts/mc_crosscheck.py)."""

import numpy as np

from s301_bayes import constants as c
from s301_bayes.catalog import load_s301
from s301_bayes.evolution import d_jj, evolve_mc, evolve_star, j_loss_cone
from s301_bayes.initial_conditions import p_j0, sample_j0


def test_initial_conditions_normalized_and_consistent():
    j = np.linspace(1e-4, 1.0, 20001)
    pj = p_j0(j, c.A_301)
    assert abs(np.trapz(pj, j) - 1.0) < 1e-4
    rng = np.random.default_rng(5)
    js = sample_j0(500_000, c.A_301, rng)
    cdf = np.cumsum(pj)
    cdf /= cdf[-1]
    for q in (0.1, 0.5, 0.9):
        assert abs(np.quantile(js, q) - j[np.searchsorted(cdf, q)]) < 2e-3


def test_fp_matches_mc_at_s301():
    s = load_s301()
    gamma = 1.1
    sol = evolve_star(s.a_au, gamma)
    n, steps = 500_000, 600
    rng = np.random.default_rng(30)
    j0 = sample_j0(n, s.a_au, rng)
    jf, alive = evolve_mc(
        j0, d_jj(s.a_au, gamma), c.T_FLIGHT, steps, rng, j_loss_cone(s.a_au)
    )

    ps_mc = alive.mean()
    ps_err = np.sqrt(ps_mc * (1.0 - ps_mc) / n)
    assert abs(ps_mc - sol["p_surv"]) < max(4.0 * ps_err, 0.005 * sol["p_surv"])

    ji = np.sqrt(1.0 - s.e**2)
    delta = 0.01
    cnt = int(np.sum(np.abs(jf[alive] - ji) < delta))
    pe_mc = cnt / (alive.sum() * 2.0 * delta) * s.e / ji
    pe_fp = sol["p_e"](s.e)
    stat = pe_mc / np.sqrt(cnt)
    assert abs(pe_mc - pe_fp) < max(4.0 * stat, 0.02 * pe_fp)


def test_fp_grid_convergence():
    sol = evolve_star(c.A_301, 1.1)
    coarse = evolve_star(c.A_301, 1.1, n_j=1201, n_t=600)
    assert abs(coarse["p_surv"] / sol["p_surv"] - 1.0) < 2e-3
    assert abs(coarse["p_e"](c.E_301) / sol["p_e"](c.E_301) - 1.0) < 3e-3


def test_survival_decreases_with_larger_absorbing_radius():
    """The absorbing boundary must be at 3 AU (not 1 AU): a larger loss cone
    absorbs more."""
    jlc_3 = j_loss_cone(c.A_301)
    x1 = 1.0 / c.A_301
    jlc_1 = np.sqrt(x1 * (2.0 - x1))
    assert jlc_3 > jlc_1
    j = np.linspace(jlc_1, 1.0, 2401)
    p0 = p_j0(j, c.A_301)
    from s301_bayes.evolution import fp_solve

    p0[0] = 0.0
    pf1 = fp_solve(j, p0, d_jj(c.A_301, 1.1), c.T_FLIGHT, 1200)
    surv_1au = np.trapz(np.maximum(pf1, 0.0), j)
    surv_3au = evolve_star(c.A_301, 1.1)["p_surv"]
    assert surv_3au < surv_1au - 0.02
