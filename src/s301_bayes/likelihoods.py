"""Orbital-parameter likelihoods under the two hypotheses.

Pair hypothesis (conditioned on surviving to the present day):

    f_pair(a, e) = u(a) * P_surv(a) * p(e | a, survived) / Z,
    Z = int u(a) P_surv(a) da   over the support  A_SUPPORT,

where u(a) = p_mb(a/C)/C is the semi-major-axis pdf induced by the
companion-mass distribution through the deterministic line a = C*m_b,
P_surv(a) and p(e|a, survived) come from the Fokker-Planck evolution.

Null hypothesis: power-law eccentricities and the density-cusp
semi-major-axis distribution,

    f_0(a, e) = (3-gamma) * a^(2-gamma) / (a_max^(3-gamma) - a_min^(3-gamma))
                * (1+alpha) * e^alpha,

normalized over the same semi-major-axis interval and e in [0, 1]. The
fiducial null is thermal, alpha = ALPHA_THERMAL = 1 (p(e) = 2e); other alpha
values are robustness checks of the null only (f_pair does not depend on
alpha), e.g. the maximum-likelihood exponent of the catalog (fit_alpha).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from . import constants as c
from ._numerics import trapezoid
from .binary_sampling import p_mb
from .evolution import evolve_star


def u_a(a_au):
    """Semi-major-axis pdf of the companion (before survival weighting),
    normalized on A_SUPPORT."""
    C = c.LINE_COEFF_AU
    return p_mb(np.asarray(a_au, dtype=float) / C) / C


class PairLikelihood:
    """f_pair(a, e) for a given density-profile slope gamma."""

    def __init__(self, gamma, n_j=2401, n_t=1200, n_a_grid=49):
        self.gamma = gamma
        self.n_j = n_j
        self.n_t = n_t
        self._solutions = {}

        # survival probability on a log-spaced grid spanning the support
        self._a_grid = np.geomspace(*c.A_SUPPORT, n_a_grid)
        ps = np.array([self._solve(a)["p_surv"] for a in self._a_grid])
        self._p_surv_spline = CubicSpline(np.log(self._a_grid), ps)

        # normalization Z = int u(a) P_surv(a) da (trapezoidal rule)
        self._a_fine = np.linspace(*c.A_SUPPORT, 4001)
        self._u_surv_fine = u_a(self._a_fine) * self.p_surv(self._a_fine)
        self.Z = self.normalization()

    def normalization(self, weight=None):
        """int u(a) P_surv(a) w(a) da over A_SUPPORT, with the same trapezoidal
        rule as Z (``weight=None``: w = 1, i.e. Z itself). ``weight`` is a
        function of an array of semi-major axes [AU]; e.g. the detection
        probability gives the normalization of the detected companions."""
        y = self._u_surv_fine
        if weight is not None:
            y = y * np.asarray(weight(self._a_fine), dtype=float)
        return float(trapezoid(y, self._a_fine))

    def _solve(self, a_au):
        key = round(float(a_au), 6)
        if key not in self._solutions:
            self._solutions[key] = evolve_star(
                a_au, self.gamma, n_j=self.n_j, n_t=self.n_t
            )
        return self._solutions[key]

    def p_surv(self, a_au):
        """Survival probability (spline-interpolated on the grid)."""
        return self._p_surv_spline(np.log(a_au))

    def __call__(self, a_au, e):
        """f_pair evaluated at a single (a, e) point.

        The e-conditional and the survival factor are computed by a dedicated
        Fokker-Planck solve at exactly this semi-major axis (no interpolation).
        """
        if not (c.A_SUPPORT[0] <= a_au <= c.A_SUPPORT[1]):
            return 0.0
        sol = self._solve(a_au)
        return float(u_a(a_au)) * sol["p_surv"] * sol["p_e"](e) / self.Z


def null_e_pdf(e, alpha=c.ALPHA_THERMAL):
    """Null eccentricity pdf p(e) = (1+alpha) e^alpha on [0, 1], alpha > -1
    (alpha = 1: thermal, p(e) = 2e)."""
    if not alpha > -1.0:
        raise ValueError(f"p(e) ~ e^alpha is not normalizable for alpha = {alpha}")
    return (1.0 + alpha) * np.asarray(e, dtype=float) ** alpha


def fit_alpha(eccentricities) -> tuple[float, float]:
    """Maximum-likelihood exponent of p(e) = (1+alpha) e^alpha for a sample of
    n eccentricities in (0, 1), and its Fisher error:
    (alpha, sigma) = (-n / sum(ln e_i) - 1, (1+alpha) / sqrt(n))."""
    e = np.asarray(eccentricities, dtype=float)
    if e.ndim != 1 or e.size == 0 or np.any((e <= 0.0) | (e >= 1.0)):
        raise ValueError("need a non-empty 1-D sample of eccentricities in (0, 1)")
    alpha = float(-e.size / np.sum(np.log(e)) - 1.0)
    return alpha, float((1.0 + alpha) / np.sqrt(e.size))


def null_likelihood(a_au, e, gamma, alpha=c.ALPHA_THERMAL):
    """f_0(a, e): eccentricity pdf (1+alpha) e^alpha times the cusp
    a-distribution. alpha = 1 (default) is the thermal null, p(e) = 2e, bit
    for bit.

    Only the null depends on alpha. The pair model is unchanged: its
    j-diffusion keeps the drift D/(2j) that makes the *thermal* distribution
    stationary, so B_i(alpha) / B_i(1) = 2 e_i^(1-alpha) / (1+alpha).
    """
    a_lo, a_hi = c.A_SUPPORT
    a_au = np.asarray(a_au, dtype=float)
    g = 3.0 - gamma
    norm = g / (a_hi**g - a_lo**g)
    out = norm * a_au ** (2.0 - gamma) * null_e_pdf(e, alpha)
    out = np.where((a_au >= a_lo) & (a_au <= a_hi), out, 0.0)
    return out if out.ndim else float(out)
