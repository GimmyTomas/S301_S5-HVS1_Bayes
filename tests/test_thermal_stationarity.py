"""The drift term D/(2j) is what makes the thermal distribution p(j) ~ j
stationary under constant-D diffusion. These tests guard against dropping it
(a walk without drift relaxes towards uniform in j, i.e. piles up at low j)."""

import numpy as np

from s301_bayes._numerics import trapezoid
from s301_bayes.evolution import fp_solve


def test_fp_thermal_is_stationary():
    jlc = 0.05
    j = np.linspace(jlc, 1.0, 1001)
    p_th = 2.0 * j / (1.0 - jlc**2)
    pf = fp_solve(j, p_th.copy(), D=3e-9, t_total=3e7, n_steps=500, bottom="reflecting")
    assert np.max(np.abs(pf - p_th) / p_th) < 1e-9
    # probability is conserved with reflecting boundaries
    assert abs(trapezoid(pf, j) - trapezoid(p_th, j)) < 1e-10


def _reflected_walk(n, t_total, n_steps, D, jlo, rng, drift):
    """SDE walk reflected at both jlo and 1, with the drift term optional."""
    u = rng.random(n)
    j = np.sqrt(jlo**2 + u * (1.0 - jlo**2))  # thermal initial sample
    dt = t_total / n_steps
    sig = np.sqrt(D * dt)
    for _ in range(n_steps):
        mu = D / (2.0 * j) * dt if drift else 0.0
        j = j + mu + sig * rng.standard_normal(n)
        j = np.where(j > 1.0, 2.0 - j, j)
        j = np.where(j < jlo, 2.0 * jlo - j, j)
    return j


def test_mc_thermal_stationary_only_with_drift():
    jlo, D, t = 0.05, 3e-9, 3e7
    mean_th = (2.0 / 3.0) * (1.0 - jlo**3) / (1.0 - jlo**2)  # thermal mean j
    n = 200_000
    tol = 5.0 * np.sqrt(0.05 / n)  # 5 sigma on the mean (var(j) ~ 0.05)

    rng = np.random.default_rng(11)
    j_with = _reflected_walk(n, t, 400, D, jlo, rng, drift=True)
    assert abs(j_with.mean() - mean_th) < tol

    rng = np.random.default_rng(11)
    j_without = _reflected_walk(n, t, 400, D, jlo, rng, drift=False)
    assert mean_th - j_without.mean() > 10.0 * tol  # piles up at low j: wrong
