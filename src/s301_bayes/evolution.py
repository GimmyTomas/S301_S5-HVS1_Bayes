"""Angular-momentum diffusion over the flight time.

The dimensionless angular momentum j = sqrt(1-e^2) performs a random walk with
constant (j-independent) diffusion coefficient
    D_jj(a) = J_301^2 / t_j(a),      t_j(a) = T_J_301 * (a/A_301)^(gamma-3/2),
supplemented by the geometric drift term
    dj = [D_jj/(2j)] dt + sqrt(D_jj) dW,
which is the unique constant-D Langevin equation whose stationary distribution
is the thermal p(j) ~ j (equivalently p(e) = 2e). Omitting the drift makes the
walk pile up at low j and is *wrong* -- guarded by tests/test_thermal_stationarity.

Boundaries: reflecting at j = 1 (e = 0), absorbing at the pericenter distance
R_ABSORB_AU (tidal disruption), i.e. at j_lc(a) = sqrt(1 - (1 - R_ABSORB/a)^2).

Two solvers of the same process are provided:
 * a deterministic Crank-Nicolson Fokker-Planck solve (primary results), in a
   flux form F = -(D/2) j d(p/j)/dj that is exactly zero on the thermal state,
   so thermality is preserved to machine precision on the grid;
 * a seeded Euler-Maruyama Monte Carlo (validation).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from . import constants as c
from .initial_conditions import p_j0


def t_j(a_au, gamma):
    """j-diffusion time [yr] at semi-major axis a."""
    return c.T_J_301 * (a_au / c.A_301) ** (gamma - 1.5)


def d_jj(a_au, gamma):
    """Diffusion coefficient D_jj [1/yr]."""
    return c.J_301**2 / t_j(a_au, gamma)


def j_loss_cone(a_au):
    """Absorbing boundary in j: pericenter a(1-e) = R_ABSORB_AU."""
    x = c.R_ABSORB_AU / a_au
    return np.sqrt(x * (2.0 - x))


# ----------------------------------------------------------------------------
# Fokker-Planck (Crank-Nicolson, finite volume)
# ----------------------------------------------------------------------------

def _fp_operator(j, D, bottom):
    """Tridiagonal generator L with dp/dt = L p; flux form
    F_{i+1/2} = -(D/2) j_{i+1/2} (p_{i+1}/j_{i+1} - p_i/j_i)/h."""
    n = j.size
    h = j[1] - j[0]
    jm = 0.5 * (j[1:] + j[:-1])          # j at cell interfaces, length n-1
    k = D / (2.0 * h**2)

    lower = np.zeros(n - 1)              # L[i, i-1]
    main = np.zeros(n)                   # L[i, i]
    upper = np.zeros(n - 1)              # L[i, i+1]

    # interior nodes
    main[1:-1] = -k * (jm[1:] + jm[:-1]) / j[1:-1]
    lower[:-1] = k * jm[:-1] / j[:-2]    # coefficient of p_{i-1} in row i
    upper[1:] = k * jm[1:] / j[2:]       # coefficient of p_{i+1} in row i

    # top boundary (j = 1): reflecting, zero flux through the upper interface
    main[-1] = -k * jm[-1] / j[-1]
    lower[-1] = k * jm[-1] / j[-2]

    if bottom == "absorbing":
        # Dirichlet p_0 = 0: freeze the boundary node; row 1 still drains
        # probability through the lower interface flux.
        main[0] = 0.0
        upper[0] = 0.0
    elif bottom == "reflecting":
        main[0] = -k * jm[0] / j[0]
        upper[0] = k * jm[0] / j[1]
    else:
        raise ValueError(bottom)
    return lower, main, upper


def fp_solve(j, p0, D, t_total, n_steps, bottom="absorbing"):
    """Crank-Nicolson evolution of p(j) for time t_total; returns p(j, t_total)."""
    lower, main, upper = _fp_operator(j, D, bottom)
    dt = t_total / n_steps
    n = j.size

    def tri(scale):
        return diags(
            [scale * lower, 1.0 + scale * main, scale * upper],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format="csc",
        )

    lhs = splu(tri(-0.5 * dt))
    rhs = tri(+0.5 * dt)
    p = p0.copy()
    if bottom == "absorbing":
        p[0] = 0.0
    for _ in range(n_steps):
        p = lhs.solve(rhs @ p)
    return p


def evolve_star(a_au, gamma, n_j=2401, n_t=1200):
    """Full forward evolution at fixed semi-major axis.

    Returns dict with the j-grid, the final (unnormalized) pdf p(j, t_flight),
    the survival probability, and a callable p_e(e) giving the normalized
    final eccentricity pdf conditioned on survival.
    """
    jlc = j_loss_cone(a_au)
    j = np.linspace(jlc, 1.0, n_j)
    p0 = p_j0(j, a_au)  # normalized over [0, 1]; mass below jlc absorbed at birth
    p0[0] = 0.0
    pf = np.maximum(fp_solve(j, p0, d_jj(a_au, gamma), c.T_FLIGHT, n_t), 0.0)
    p_surv = float(np.trapz(pf, j))

    def p_e(e):
        e = np.asarray(e, dtype=float)
        jj = np.sqrt(np.clip(1.0 - e**2, 0.0, 1.0))
        pj = np.interp(jj, j, pf, left=0.0, right=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = pj * np.where(jj > 0, e / jj, 0.0) / p_surv
        return out if e.ndim else float(out)

    return {"j": j, "p_final": pf, "p_surv": p_surv, "p_e": p_e, "a_au": a_au}


# ----------------------------------------------------------------------------
# Euler-Maruyama Monte Carlo (validation)
# ----------------------------------------------------------------------------

def evolve_mc(j0, D, t_total, n_steps, rng, j_lc, bridge=True):
    """Evolve samples j0 (with per-sample D and j_lc allowed); returns the
    final j values and the survival mask.

    With ``bridge=True`` (default), absorption is supplemented by the
    Brownian-bridge probability exp(-2 (j_old-b)(j_new-b) / (D dt)) of having
    crossed the boundary *inside* a step, which removes the O(sqrt(dt))
    survival bias of the naive discretely-checked walk.
    """
    j = np.array(j0, dtype=float)
    D = np.broadcast_to(np.asarray(D, dtype=float), j.shape)
    j_lc = np.broadcast_to(np.asarray(j_lc, dtype=float), j.shape)
    dt = t_total / n_steps
    sig = np.sqrt(D * dt)
    alive = j > j_lc
    for _ in range(n_steps):
        idx = np.flatnonzero(alive)
        jo = j[idx]
        b = j_lc[idx]
        step = D[idx] / (2.0 * jo) * dt + sig[idx] * rng.standard_normal(idx.size)
        jn = jo + step
        jn = np.where(jn > 1.0, 2.0 - jn, jn)  # reflect at j = 1
        dead = jn <= b
        if bridge:
            p_cross = np.exp(
                -2.0 * np.maximum(jo - b, 0.0) * np.maximum(jn - b, 0.0)
                / (D[idx] * dt)
            )
            dead |= rng.random(idx.size) < p_cross
        j[idx] = jn
        alive[idx[dead]] = False
    return j, alive
