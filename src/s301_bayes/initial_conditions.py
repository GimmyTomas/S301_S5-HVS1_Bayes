"""Initial (capture) angular-momentum distribution.

The initial eccentricity follows the capture relation
    1 - e0 = kappa * (m_bin / M_BH)^(1/3),      m_bin = m_ej + m_b,
with kappa lognormal (median KAPPA_MEDIAN, sigma_ln = KAPPA_SIGMA_LN).
At fixed semi-major axis a, the companion mass is m_b = a / LINE_COEFF_AU and
the ejected mass m_ej is marginalized over its conditional distribution
(Gaussian mass likelihood times the binary population weights).

Densities are expressed in the dimensionless angular momentum j = sqrt(1-e^2);
with x = 1 - e, the Jacobian is p_j(j) = p_x(x) * j / (1 - x).
"""

import numpy as np

from . import constants as c
from ._numerics import trapezoid
from .binary_sampling import p_me_given_mb

_ME_GRID = np.linspace(*c.M_EJ_RANGE, 241)
_MU_LN = np.log(c.KAPPA_MEDIAN)


def _lognorm_pdf(x):
    x = np.maximum(x, 1e-300)
    s = c.KAPPA_SIGMA_LN
    return np.exp(-((np.log(x) - _MU_LN) ** 2) / (2 * s**2)) / (
        x * s * np.sqrt(2 * np.pi)
    )


def p_j0(j, a_au):
    """Initial pdf of j at fixed semi-major axis (normalized over j in [0,1],
    up to a totally negligible lognormal tail at 1 - e0 > 1)."""
    j = np.asarray(j, dtype=float)
    mb = a_au / c.LINE_COEFF_AU
    weights = p_me_given_mb(_ME_GRID, mb)
    cbrt = ((mb + _ME_GRID) / c.M_BH) ** (1.0 / 3.0)

    e = np.sqrt(np.clip(1.0 - j**2, 0.0, 1.0))
    x = 1.0 - e  # = 1 - e0
    # p_x(x) = int dm_e p(m_e|m_b) LN(x / c(m_e)) / c(m_e)
    px = trapezoid(
        weights[None, :] * _lognorm_pdf(x[..., None] / cbrt) / cbrt,
        _ME_GRID,
        axis=-1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(e > 0.0, j / e, np.inf)
    return np.where((j > 0.0) & (j < 1.0), px * np.nan_to_num(jac, posinf=0.0), 0.0)


def sample_j0(n, a_au, rng):
    """Monte Carlo draw of initial j at fixed a (for validation)."""
    mb = a_au / c.LINE_COEFF_AU
    weights = p_me_given_mb(_ME_GRID, mb)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    me = np.interp(rng.random(n), cdf, _ME_GRID)
    kappa = np.exp(rng.normal(_MU_LN, c.KAPPA_SIGMA_LN, n))
    x = kappa * ((mb + me) / c.M_BH) ** (1.0 / 3.0)
    x = np.minimum(x, 1.0)
    return np.sqrt(x * (2.0 - x))
