"""Orbital-parameter likelihoods under the two hypotheses.

Pair hypothesis (conditioned on surviving to the present day):

    f_pair(a, e) = u(a) * P_surv(a) * p(e | a, survived) / Z,
    Z = int u(a) P_surv(a) da   over the support  A_SUPPORT,

where u(a) = p_mb(a/C)/C is the semi-major-axis pdf induced by the
companion-mass distribution through the deterministic line a = C*m_b,
P_surv(a) and p(e|a, survived) come from the Fokker-Planck evolution.

Null hypothesis: thermal eccentricities and the density-cusp semi-major-axis
distribution,

    f_0(a, e) = (3-gamma) * a^(2-gamma) / (a_max^(3-gamma) - a_min^(3-gamma)) * 2e,

normalized over the same semi-major-axis interval.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from . import constants as c
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

        # normalization Z = int u(a) P_surv(a) da
        a_fine = np.linspace(*c.A_SUPPORT, 4001)
        self.Z = float(np.trapz(u_a(a_fine) * self.p_surv(a_fine), a_fine))

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


def null_likelihood(a_au, e, gamma):
    """f_0(a, e): thermal eccentricity times the cusp a-distribution."""
    a_lo, a_hi = c.A_SUPPORT
    a_au = np.asarray(a_au, dtype=float)
    g = 3.0 - gamma
    norm = g / (a_hi**g - a_lo**g)
    out = norm * a_au ** (2.0 - gamma) * 2.0 * np.asarray(e, dtype=float)
    out = np.where((a_au >= a_lo) & (a_au <= a_hi), out, 0.0)
    return out if out.ndim else float(out)
