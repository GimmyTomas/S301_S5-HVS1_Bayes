"""Population model of the disrupted binary and the companion-mass distribution.

The binary is drawn as: primary mass m1 from the Kroupa (2001) IMF (only the
m > 0.5 Msun branch, slope -2.3, is ever relevant), secondary m2 = q*m1 with
p(q) ~ q^-0.3 on [0.1, 1] plus a 10% twin excess uniform on [0.9, 1].
Each star of the binary is the ejected one with equal prior probability 1/2,
and the pair is weighted by the Gaussian measurement likelihood of S5-HVS1's
mass for the ejected star, m_ej ~ 2.35 +- 0.06 Msun. The captured companion
m_b is the other member, restricted to [1.1, 7] Msun.

The companion-mass density p(m_b) is computed by direct quadrature of the two
ordered configurations,

  p(m_b) propto 1/2 [ A(m_b) + B(m_b) ],
  A(m_b) = int dm  xi(m)  w(m)   phi(m_b/m) / m ,   m >= m_b   (m_b = secondary),
  B(m_b) = xi(m_b)/m_b * int dm_e w(m_e) phi(m_e/m_b) ,  m_e <= m_b  (m_b = primary),

with xi the IMF, phi the mass-ratio pdf and w the Gaussian mass likelihood
(truncated at 8 sigma, where it is < 1e-14). Both ordered terms always carry
weight 1/2: the equal ejection prior is what makes the treatment of
nearly-equal-mass binaries correct. The constraint q >= 0.1 never binds for
m_b in [1.1, 7] Msun.
"""

import numpy as np

from . import constants as c

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def imf(m):
    """Kroupa IMF, unnormalized, valid for m > 0.5 Msun (slope -2.3)."""
    return np.asarray(m, dtype=float) ** (-c.IMF_ALPHA)


def mass_weight(m):
    """Gaussian measurement likelihood of the ejected mass (peak-normalized)."""
    m = np.asarray(m, dtype=float)
    return np.exp(-((m - c.M_EJ) ** 2) / (2.0 * c.M_EJ_SIGMA**2))


# normalization of the power-law part of p(q) on [Q_MIN, 1]
_PL_EXP = 1.0 + c.Q_ALPHA                          # = 0.7
_PL_NORM = _PL_EXP / (1.0 - c.Q_MIN**_PL_EXP)


def q_pdf(q):
    """Mass-ratio pdf on [Q_MIN, 1]: power law plus uniform twin excess."""
    q = np.asarray(q, dtype=float)
    pl = (1.0 - c.F_TWIN) * _PL_NORM * q**c.Q_ALPHA
    tw = c.F_TWIN / (1.0 - c.Q_TWIN_MIN) * (q >= c.Q_TWIN_MIN)
    return np.where((q >= c.Q_MIN) & (q <= 1.0), pl + tw, 0.0)


def _integrate(f, lo, hi, breaks=()):
    """Gauss-Legendre integration of f on [lo, hi], split at interior breaks
    (the q-pdf is discontinuous at q = Q_TWIN_MIN)."""
    if hi <= lo:
        return 0.0
    pts = np.array([lo, hi] + [b for b in breaks if lo < b < hi])
    pts.sort()
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        x = 0.5 * (b - a) * _GL_NODES + 0.5 * (a + b)
        total += 0.5 * (b - a) * np.dot(_GL_WEIGHTS, f(x))
    return total


def _case_A(mb):
    """m_b is the secondary; the ejected star is the primary m >= m_b."""
    lo = max(mb, c.M_EJ_RANGE[0])
    # discontinuity of phi at q = m_b/m = Q_TWIN_MIN  ->  m = m_b/Q_TWIN_MIN
    return _integrate(
        lambda m: imf(m) * mass_weight(m) * q_pdf(mb / m) / m,
        lo,
        c.M_EJ_RANGE[1],
        breaks=(mb / c.Q_TWIN_MIN,),
    )


def _case_B(mb):
    """m_b is the primary; the ejected star is the secondary m_e <= m_b."""
    hi = min(mb, c.M_EJ_RANGE[1])
    # discontinuity of phi at q = m_e/m_b = Q_TWIN_MIN  ->  m_e = Q_TWIN_MIN*m_b
    return float(imf(mb)) / mb * _integrate(
        lambda me: mass_weight(me) * q_pdf(me / mb),
        c.M_EJ_RANGE[0],
        hi,
        breaks=(c.Q_TWIN_MIN * mb,),
    )


def p_mb_unnorm(mb):
    """Unnormalized companion-mass density (scalar or array argument)."""
    mb_arr = np.atleast_1d(np.asarray(mb, dtype=float))
    out = np.array([0.5 * (_case_A(m) + _case_B(m)) for m in mb_arr])
    out[(mb_arr < c.MB_MIN) | (mb_arr > c.MB_MAX)] = 0.0
    return out if np.ndim(mb) else float(out[0])


_NORM_CACHE = {}


def p_mb_norm() -> float:
    """Normalization integral of p_mb_unnorm over [MB_MIN, MB_MAX]."""
    if "norm" not in _NORM_CACHE:
        grid = np.linspace(c.MB_MIN, c.MB_MAX, 4001)
        _NORM_CACHE["norm"] = float(np.trapz(p_mb_unnorm(grid), grid))
    return _NORM_CACHE["norm"]


def p_mb(mb):
    """Normalized companion-mass pdf on [MB_MIN, MB_MAX]."""
    return p_mb_unnorm(mb) / p_mb_norm()


def p_me_given_mb(me_grid, mb):
    """Conditional pdf of the ejected mass given the companion mass.
    Normalized on ``me_grid`` (which must span M_EJ_RANGE)."""
    me = np.asarray(me_grid, dtype=float)
    dens = 0.5 * mass_weight(me) * (
        np.where(me >= mb, imf(me) * q_pdf(mb / np.maximum(me, mb)) / me, 0.0)
        + np.where(me <= mb, imf(mb) * q_pdf(me / mb) / mb, 0.0)
    )
    norm = np.trapz(dens, me)
    return dens / norm


# ----------------------------------------------------------------------------
# Brute-force Monte Carlo sampler (used for validation)
# ----------------------------------------------------------------------------

def sample_q(n, rng):
    """Draw mass ratios from the power-law + twin-excess mixture."""
    q = np.empty(n)
    twin = rng.random(n) < c.F_TWIN
    q[twin] = rng.uniform(c.Q_TWIN_MIN, 1.0, twin.sum())
    u = rng.random(n - twin.sum())
    q[~twin] = (c.Q_MIN**_PL_EXP + u * (1.0 - c.Q_MIN**_PL_EXP)) ** (1.0 / _PL_EXP)
    return q


def sample_primary(n, rng, m_lo, m_hi):
    """Draw primaries from the m^-2.3 IMF truncated to [m_lo, m_hi]."""
    g = 1.0 - c.IMF_ALPHA  # = -1.3
    u = rng.random(n)
    return (m_lo**g + u * (m_hi**g - m_lo**g)) ** (1.0 / g)


def sample_companions(n_accepted, rng, chunk=1_000_000):
    """Sample accepted (m_b, m_ej) pairs by brute force.

    The Gaussian mass likelihood is applied by rejection (it is
    peak-normalized, so it is directly an acceptance probability), yielding
    unweighted samples. Primaries are drawn on [M_EJ_RANGE[0], MB_MAX]: any
    pair with nonzero weight has its primary in that range (the primary is
    the heavier star, so it is >= the ejected mass truncation on one side,
    and a primary > MB_MAX would fail either the companion cut or carry zero
    mass weight). The truncation is therefore an exact restriction of the
    sample space, up to the < 1e-14 tail of the truncated Gaussian.
    """
    mb_out, me_out = [], []
    n_got = 0
    while n_got < n_accepted:
        m1 = sample_primary(chunk, rng, c.M_EJ_RANGE[0], c.MB_MAX)
        m2 = sample_q(chunk, rng) * m1
        first_ejected = rng.random(chunk) < 0.5
        m_ej = np.where(first_ejected, m1, m2)
        m_b = np.where(first_ejected, m2, m1)
        ok = (
            (rng.random(chunk) < mass_weight(m_ej))
            & (m_b >= c.MB_MIN)
            & (m_b <= c.MB_MAX)
        )
        mb_out.append(m_b[ok])
        me_out.append(m_ej[ok])
        n_got += int(ok.sum())
    return np.concatenate(mb_out)[:n_accepted], np.concatenate(me_out)[:n_accepted]
