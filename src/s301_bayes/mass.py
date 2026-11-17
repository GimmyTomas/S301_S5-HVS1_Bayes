"""Mass robustness check: each star's K magnitude as an extra observable.

The fiducial Bayes factors use the orbit only, B_i = f_pair(a_i, e_i) /
f_0(a_i, e_i). The mass check adds the observed K magnitude K_i of every star
(data/photometry.csv) and is formulated in magnitude space:

    B_i^mass = B_i * (Z / Z_s) * L_pair(K_i | a_i) / p_0(K_i | det).

Mass-magnitude relation, the same for both hypotheses:

    K(m) = M_Ks(m) + DM + A_K,    DM = 5 log10(R0 / 10 pc),

with M_Ks linear in log10 m between the rows of Mamajek's main-sequence dwarf
table that have both columns (sorted by mass, 0.075-20.2 Msun). The observed
magnitude is Gaussian about K(m) with width sigma_K, and a star is detected if
K <= K_lim, under both hypotheses; Phi denotes the standard normal cdf.

Pair. The companion's true mass lies on the energy line, within its fractional
width s = LINE_SIGMA_FRAC: m ~ N(m_0, s m_0), m_0 = a/C, C = LINE_COEFF_AU, and

    L_pair(K | a) = int dm N(m; m_0, s m_0) N(K; K(m), sigma_K).

The companion-mass prior is the fiducial one (m_b in [MB_MIN, MB_MAX], i.e.
u(a) on A_SUPPORT), now also conditioned on detection. With the fiducial
normalization Z = int da u(a) P_surv(a) and

    Z_s = int da u(a) P_surv(a) P_det(a/C),
    P_det(m) = Phi((K_lim - K(m)) / sigma_K),

the density of (a, e, K) of a detected companion is
f_pair(a, e) (Z / Z_s) L_pair(K | a). The line width is neglected inside
P_det: averaging P_det over N(m; m_0, s m_0), i.e. using the exact
int_{K <= K_lim} L_pair(K | a) dK, would raise Z / Z_s by a relative 5e-4
(7e-4 with the MIST relation).

Null. Masses are independent of the orbit and follow the Kroupa IMF
xi(m) ~ m^-IMF_ALPHA on M_NULL_RANGE; normalized over the detected stars,

    p_0(K | det) = int dm xi(m) N(K; K(m), sigma_K)
                   / int dm xi(m) Phi((K_lim - K(m)) / sigma_K),   K <= K_lim.

All 18 stars use the same recipe, including S301 and the late-type giants
S17, S21 and S38, for which a main-sequence relation is only formal. For the
pair hypothesis only, a MIST v1.2 isochrone coeval with S5-HVS1 can replace
Mamajek's table (pair_relation='mist'): v/v_crit = 0.4, [Fe/H] = +0.25,
log10 age = 7.7, 2MASS Ks, the grid point closest to S5-HVS1 (log10 age 7.72,
[Fe/H] = 0.29 +- 0.08, Koposov et al. 2020).

The photometric mass m_hat (the inverse relation, photometric_mass) is only
displayed; it does not enter the Bayes factors.

Forecast (spectroscopic_forecast). A future spectroscopic mass m_spec of a
star, Gaussian about its true mass with width sigma_m, is one more observable
next to K. Both magnitude-space integrals acquire the factor
N(m_spec; m, sigma_m),

    L_pair(K, m_spec | a) = int dm N(m; m_0, s m_0) N(K; K(m), sigma_K)
                                   N(m_spec; m, sigma_m),

and likewise p_0(K, m_spec | det), so that B_i^mass is multiplied by the ratio
of the conditional densities of m_spec given K under the two hypotheses,

    [L_pair(K, m_spec | a) / L_pair(K | a)] / [p_0(K, m_spec | det) / p_0(K | det)].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.special import ndtr

from . import constants as c
from .bayes import CatalogResult, StarResult, compute_catalog
from .catalog import data_dir, load_s301, read_csv
from .likelihoods import PairLikelihood

# ----------------------------------------------------------------------------
# Input data
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Photometry:
    """One row of data/photometry.csv."""

    name: str
    k_mag: float    # observed K-band magnitude
    sptype: str     # 'e' early-type, 'l' late-type, '-' not classified
    source: str


@lru_cache(maxsize=None)
def _load_photometry(path: str) -> Mapping[str, Photometry]:
    # read-only view: the cached mapping is shared by every caller
    return MappingProxyType({
        r["star"]: Photometry(r["star"], float(r["mK"]), r["sptype"], r["source"])
        for r in read_csv(path)
    })


def load_photometry() -> Mapping[str, Photometry]:
    """{star name: Photometry} from data/photometry.csv (read-only)."""
    return _load_photometry(str(data_dir() / "photometry.csv"))


# ----------------------------------------------------------------------------
# Mass-magnitude relations
# ----------------------------------------------------------------------------


def distance_modulus(r0_pc: float = c.R0_PHOT_PC) -> float:
    """DM = 5 log10(R0 / 10 pc)."""
    return 5.0 * math.log10(r0_pc / 10.0)


@dataclass(frozen=True, eq=False)
class MassMagnitudeRelation:
    """Absolute Ks magnitude as a function of mass: M_Ks linear in log10 m
    between the nodes, constant beyond the end nodes."""

    name: str
    log_m: np.ndarray     # log10(m / Msun) at the nodes, strictly increasing
    abs_mag: np.ndarray   # M_Ks at the nodes [mag]

    @property
    def mass_range(self) -> tuple[float, float]:
        """Masses [Msun] of the first and last node."""
        return float(10.0 ** self.log_m[0]), float(10.0 ** self.log_m[-1])

    def abs_mag_k(self, m):
        """M_Ks(m) [mag] for masses m [Msun] (scalar or array)."""
        out = np.interp(np.log10(m), self.log_m, self.abs_mag)
        return out if np.ndim(out) else float(out)

    def k_mag(self, m, a_k: float = c.A_K_GC, r0_pc: float = c.R0_PHOT_PC):
        """Observed K(m) = M_Ks(m) + DM + A_K [mag]."""
        return self.abs_mag_k(m) + distance_modulus(r0_pc) + a_k


# name: (file in data/, mass column); both files also have the column M_Ks
RELATION_FILES = {
    "mamajek": ("mamajek_dwarf_MKs_mass.csv", "Msun"),
    "mist": ("mist_v1.2_feh+0.25_logage7.7_MKs.csv", "initial_mass"),
}


@lru_cache(maxsize=None)
def _load_relation(name: str, path: str, mass_col: str) -> MassMagnitudeRelation:
    rows = [r for r in read_csv(path) if r[mass_col] and r["M_Ks"]]
    m = np.array([float(r[mass_col]) for r in rows])
    abs_mag = np.array([float(r["M_Ks"]) for r in rows])
    order = np.argsort(m, kind="stable")
    log_m, abs_mag = np.log10(m[order]), abs_mag[order]
    if np.any(np.diff(log_m) <= 0.0):
        raise ValueError(f"repeated masses in {path}")
    log_m.setflags(write=False)
    abs_mag.setflags(write=False)
    return MassMagnitudeRelation(name, log_m, abs_mag)


def mass_magnitude_relation(name: str = "mamajek") -> MassMagnitudeRelation:
    """'mamajek': the rows of Mamajek's dwarf table with both mass and M_Ks
    (fiducial, both hypotheses); 'mist': the MIST v1.2 isochrone with
    [Fe/H] = +0.25, log10 age = 7.7 (pair-relation variant)."""
    if name not in RELATION_FILES:
        raise ValueError(f"unknown relation {name!r}; choose from {sorted(RELATION_FILES)}")
    fname, mass_col = RELATION_FILES[name]
    return _load_relation(name, str(data_dir() / fname), mass_col)


def photometric_mass(k_mag, a_k: float = c.A_K_GC, r0_pc: float = c.R0_PHOT_PC):
    """Main-sequence mass m_hat [Msun] of a star of observed K magnitude, for
    display only: log10 M linear in M_Ks = K - DM - A_K between the rows of
    Mamajek's table sorted by M_Ks, end segments extrapolated linearly.

    It inverts mass_magnitude_relation('mamajek') exactly except for
    1.17 < M_Ks < 1.66 (A2V-A6V, 1.83-1.98 Msun), where the tabulated masses
    are not monotonic in M_Ks."""
    rel = mass_magnitude_relation("mamajek")
    order = np.argsort(rel.abs_mag)
    mk, logm = rel.abs_mag[order], rel.log_m[order]
    x = np.asarray(k_mag, dtype=float) - distance_modulus(r0_pc) - a_k
    y = np.interp(x, mk, logm)
    s_lo = (logm[1] - logm[0]) / (mk[1] - mk[0])
    s_hi = (logm[-1] - logm[-2]) / (mk[-1] - mk[-2])
    y = np.where(x < mk[0], logm[0] + s_lo * (x - mk[0]), y)
    y = np.where(x > mk[-1], logm[-1] + s_hi * (x - mk[-1]), y)
    out = 10.0**y
    return out if out.ndim else float(out)


# ----------------------------------------------------------------------------
# Densities of K under the two hypotheses
# ----------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_LN10 = math.log(10.0)

# Mass integrals are done in x = log10 m by composite Gauss-Legendre
# quadrature, with panel edges at the nodes of the relation (K(m) is linear in
# x between nodes, so every panel integrand is smooth). The line Gaussian is
# truncated at +-_N_SIGMA_LINE sigma (weight outside 1.2e-15), with panels at
# most one line sigma wide; null panels are at most _DX_NULL wide, across which
# K varies by at most 0.42 mag (the steep A4V-A2V segment; < 0.21 mag elsewhere).
# tests/test_mass.py checks every integral against scipy.integrate.quad.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_N_SIGMA_LINE = 8.0
_DX_NULL = 0.02   # dex


def _gauss(x, mu, s):
    """Normalized Gaussian density N(x; mu, s)."""
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * _SQRT_2PI)


def _composite_gl(lo, hi, breaks, max_width):
    """Nodes and weights of composite Gauss-Legendre quadrature on [lo, hi],
    with panel edges at every break inside (lo, hi) and panels at most
    max_width wide."""
    inner = breaks[(breaks > lo) & (breaks < hi)]
    edges = np.concatenate(([lo], inner, [hi]))
    n_sub = np.maximum(1, np.ceil(np.diff(edges) / max_width)).astype(int)
    edges = np.concatenate(
        [np.linspace(a, b, n + 1)[:-1] for a, b, n in zip(edges[:-1], edges[1:], n_sub)]
        + [[hi]]
    )
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    return (mid + half * _GL_X).ravel(), (half * _GL_W).ravel()


def imf_pdf(m, m_range: tuple[float, float] = c.M_NULL_RANGE):
    """Null mass function: Kroupa xi(m) ~ m^-IMF_ALPHA normalized to 1 on
    m_range, zero outside."""
    lo, hi = m_range
    a = c.IMF_ALPHA
    m = np.asarray(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - 1.0) * m ** (-a) / (lo ** (1.0 - a) - hi ** (1.0 - a))
    out = np.where((m >= lo) & (m <= hi), out, 0.0)
    return out if out.ndim else float(out)


@dataclass(frozen=True)
class MassModel:
    """Settings of the mass check and the densities of K it implies.

    sigma_k        Gaussian scatter of K about K(m) [mag]
    k_lim          detection limit, K <= k_lim [mag] (both hypotheses)
    a_k            K-band extinction [mag], in K(m) = M_Ks(m) + DM + A_K
    line_sigma     fractional width of the energy line (0: m = a/C exactly)
    pair_relation  M_Ks(m) of the pair hypothesis, 'mamajek' or 'mist'
                   (the null always uses 'mamajek')
    m_null         support of the null Kroupa mass function [Msun]
    """

    sigma_k: float = c.SIGMA_K
    k_lim: float = c.K_LIM
    a_k: float = c.A_K_GC
    line_sigma: float = c.LINE_SIGMA_FRAC
    pair_relation: str = "mamajek"
    m_null: tuple[float, float] = c.M_NULL_RANGE

    def __post_init__(self):
        if not self.sigma_k > 0.0:
            raise ValueError("sigma_k must be positive")
        if not 0.0 <= self.line_sigma * _N_SIGMA_LINE < 1.0:
            raise ValueError(f"line_sigma must lie in [0, {1.0 / _N_SIGMA_LINE})")
        mass_magnitude_relation(self.pair_relation)  # raises on an unknown name
        lo, hi = self.m_null
        nodes = self.null_rel.log_m
        if not (0.0 < lo < hi and nodes[0] <= np.log10(lo) and np.log10(hi) <= nodes[-1]):
            t_lo, t_hi = self.null_rel.mass_range
            raise ValueError(
                f"m_null must lie inside Mamajek's table, [{t_lo:g}, {t_hi:g}] Msun"
            )

    @property
    def pair_rel(self) -> MassMagnitudeRelation:
        """Mass-magnitude relation of the pair hypothesis."""
        return mass_magnitude_relation(self.pair_relation)

    @property
    def null_rel(self) -> MassMagnitudeRelation:
        """Mass-magnitude relation of the null hypothesis (Mamajek's table)."""
        return mass_magnitude_relation("mamajek")

    def k_pair(self, m):
        """K(m) of the pair hypothesis [mag]."""
        return self.pair_rel.k_mag(m, self.a_k)

    def k_null(self, m):
        """K(m) of the null hypothesis [mag]."""
        return self.null_rel.k_mag(m, self.a_k)

    # --- pair -----------------------------------------------------------------

    def _line_nodes(self, a_au: float, max_frac: float = math.inf):
        """Masses m_j and weights w_j with sum_j w_j g(m_j) =
        int dm N(m; m_0, s m_0) g(m), m_0 = a/C (a single node if s = 0).
        max_frac: upper limit on the fractional panel width, for a g that
        varies faster than the line Gaussian."""
        m0 = a_au / c.LINE_COEFF_AU
        s = self.line_sigma
        if s == 0.0:
            return np.array([m0]), np.array([1.0])
        x, w = _composite_gl(
            math.log10(m0 * (1.0 - _N_SIGMA_LINE * s)),
            math.log10(m0 * (1.0 + _N_SIGMA_LINE * s)),
            self.pair_rel.log_m,
            min(s, max_frac) / _LN10,
        )
        m = 10.0**x
        return m, w * _LN10 * m * _gauss(m, m0, s * m0)

    def pair_density(self, k, a_au: float):
        """L_pair(K | a) [1/mag], normalized over all K."""
        m, w = self._line_nodes(a_au)
        k = np.asarray(k, dtype=float)
        out = _gauss(k[..., None], self.k_pair(m), self.sigma_k) @ w
        return out if out.ndim else float(out)

    def pair_density_spec(self, k: float, a_au: float, m_spec: float, sigma_m: float) -> float:
        """L_pair(K, m_spec | a) [1/mag/Msun]: joint density of K and of a
        measured mass m_spec ~ N(m, sigma_m), normalized over all (K, m_spec)."""
        m, w = self._line_nodes(a_au, sigma_m / m_spec)
        return float(
            (_gauss(k, self.k_pair(m), self.sigma_k) * _gauss(m_spec, m, sigma_m)) @ w
        )

    def pair_detection(self, a_au):
        """Detection probability of a companion at a, P_det(a/C) =
        Phi((k_lim - K(a/C)) / sigma_K) (line width neglected)."""
        m0 = np.asarray(a_au, dtype=float) / c.LINE_COEFF_AU
        out = ndtr((self.k_lim - self.k_pair(m0)) / self.sigma_k)
        return out if np.ndim(out) else float(out)

    def selection_ratio(self, pair: PairLikelihood) -> float:
        """Z / Z_s: the fiducial normalization of f_pair over the one of the
        detected companions (same trapezoidal rule, PairLikelihood.normalization)."""
        return pair.Z / pair.normalization(self.pair_detection)

    # --- null -----------------------------------------------------------------

    def _null_nodes(self, max_frac: float = math.inf):
        """Masses m_j and weights w_j with sum_j w_j g(m_j) = int dm xi(m) g(m).
        max_frac: upper limit on the fractional panel width (default: _DX_NULL
        dex wide panels)."""
        lo, hi = self.m_null
        x, w = _composite_gl(
            math.log10(lo), math.log10(hi), self.null_rel.log_m,
            min(_DX_NULL, max_frac / _LN10),
        )
        m = 10.0**x
        return m, w * _LN10 * m * imf_pdf(m, self.m_null)

    def null_detection(self, k_lim: float | None = None) -> float:
        """Detected fraction of the null population,
        int dm xi(m) Phi((k_lim - K(m)) / sigma_K) (k_lim: default self.k_lim)."""
        k_lim = self.k_lim if k_lim is None else k_lim
        m, w = self._null_nodes()
        return float(ndtr((k_lim - self.k_null(m)) / self.sigma_k) @ w)

    def null_density(self, k, k_lim: float | None = None):
        """p_0(K | det) [1/mag]: normalized over K <= k_lim (default
        self.k_lim), zero above."""
        k_lim = self.k_lim if k_lim is None else k_lim
        m, w = self._null_nodes()
        k = np.asarray(k, dtype=float)
        p0 = _gauss(k[..., None], self.k_null(m), self.sigma_k) @ w
        out = np.where(k <= k_lim, p0 / self.null_detection(k_lim), 0.0)
        return out if out.ndim else float(out)

    def null_density_spec(
        self, k: float, m_spec: float, sigma_m: float, k_lim: float | None = None
    ) -> float:
        """p_0(K, m_spec | det) [1/mag/Msun]: joint density of K and of a
        measured mass m_spec ~ N(m, sigma_m), normalized over K <= k_lim
        (default self.k_lim) and all m_spec; zero for K > k_lim."""
        k_lim = self.k_lim if k_lim is None else k_lim
        if k > k_lim:
            return 0.0
        m, w = self._null_nodes(sigma_m / m_spec)
        p0 = (_gauss(k, self.k_null(m), self.sigma_k) * _gauss(m_spec, m, sigma_m)) @ w
        return float(p0 / self.null_detection(k_lim))


# ----------------------------------------------------------------------------
# Catalog with K magnitudes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MassStarResult(StarResult):
    """A catalog star with its K magnitude. f_pair, f_null: densities of
    (a, e, K) of a detected star; bayes = B_i * factor."""

    orbital: StarResult   # orbit-only result: B_i = orbital.bayes
    k_mag: float          # observed K [mag]
    sptype: str           # 'e' early-type, 'l' late-type, '-' not classified
    m_hat: float          # photometric mass [Msun], display only
    k_line: float         # K(a/C) of the pair relation [mag]
    l_pair: float         # L_pair(K | a) [1/mag]
    p_null: float         # p_0(K | det) [1/mag]
    z_ratio: float        # Z / Z_s
    k_lim_null: float     # detection limit of the null normalization [mag]

    @property
    def m_line(self) -> float:
        """Line mass a/C [Msun]."""
        return self.star.a_au / c.LINE_COEFF_AU

    @property
    def factor(self) -> float:
        """B_i^mass / B_i = (Z / Z_s) L_pair(K | a) / p_0(K | det)."""
        return self.z_ratio * self.l_pair / self.p_null

    @property
    def bayes_orbital(self) -> float:
        """Orbit-only Bayes factor B_i."""
        return self.orbital.bayes

    @property
    def bayes(self) -> float:
        """B_i^mass = B_i * factor."""
        return self.orbital.bayes * self.factor


@dataclass(frozen=True)
class MassCatalogResult(CatalogResult):
    """CatalogResult with B_i -> B_i^mass: ``results`` are MassStarResult,
    ``mirror_b301`` includes the mass factor, ``orbital`` is the orbit-only
    CatalogResult."""

    model: MassModel = field(default_factory=MassModel)
    z_ratio: float = 1.0   # Z / Z_s
    k_lim_star: Mapping[str, float] = field(default_factory=dict)
    orbital: CatalogResult | None = None


def compute_catalog_mass(
    gamma,
    pair: PairLikelihood | None = None,
    alpha: float = c.ALPHA_THERMAL,
    *,
    sigma_k: float = c.SIGMA_K,
    k_lim: float = c.K_LIM,
    a_k: float = c.A_K_GC,
    line_sigma: float = c.LINE_SIGMA_FRAC,
    pair_relation: str = "mamajek",
    m_null: tuple[float, float] = c.M_NULL_RANGE,
    k_lim_star: Mapping[str, float] | None = None,
) -> MassCatalogResult:
    """Bayes factors B_i^mass of the catalog (orbital part from compute_catalog).

    sigma_k, k_lim, a_k, line_sigma, pair_relation, m_null: see MassModel.
    k_lim_star: {name: K_lim} replacing k_lim in the null normalization of the
    named stars only (the detection model of the pair hypothesis keeps k_lim).
    """
    model = MassModel(sigma_k, k_lim, a_k, line_sigma, pair_relation, tuple(m_null))
    k_lim_star = dict(k_lim_star or {})
    pair = pair or PairLikelihood(gamma)
    base = compute_catalog(gamma, pair, alpha)
    unknown = set(k_lim_star) - {r.star.name for r in base.results}
    if unknown:
        raise KeyError(f"not in the catalog: {sorted(unknown)}")
    z_ratio = model.selection_ratio(pair)
    phot = load_photometry()

    def densities(name, a_au):
        """(L_pair(K | a), p_0(K | det), null detection limit) of a star."""
        k = phot[name].k_mag
        lim = k_lim_star.get(name, k_lim)
        if k > min(lim, k_lim):
            raise ValueError(
                f"{name}: K = {k} is fainter than the detection limit {min(lim, k_lim)}"
            )
        return model.pair_density(k, a_au), model.null_density(k, lim), lim

    results = []
    for r in base.results:
        name = r.star.name
        l_pair, p_null, lim = densities(name, r.star.a_au)
        results.append(
            MassStarResult(
                star=r.star,
                f_pair=r.f_pair * z_ratio * l_pair,
                f_null=r.f_null * p_null,
                orbital=r,
                k_mag=phot[name].k_mag,
                sptype=phot[name].sptype,
                m_hat=photometric_mass(phot[name].k_mag, a_k),
                k_line=model.k_pair(r.star.a_au / c.LINE_COEFF_AU),
                l_pair=l_pair,
                p_null=p_null,
                z_ratio=z_ratio,
                k_lim_null=lim,
            )
        )
    l_pair, p_null, _ = densities("S301", load_s301("mirror").a_au)
    return MassCatalogResult(
        gamma=gamma,
        results=results,
        mirror_b301=base.mirror_b301 * z_ratio * l_pair / p_null,
        alpha=alpha,
        model=model,
        z_ratio=z_ratio,
        k_lim_star=k_lim_star,
        orbital=base,
    )


# ----------------------------------------------------------------------------
# Forecast: a spectroscopic mass measurement
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectroscopicForecast:
    """Bayes factors if one star had a measured mass m_spec +- sigma_m in
    addition to its K magnitude (all other stars as in the mass check)."""

    name: str           # the star with the measurement
    m_spec: float       # measured mass [Msun]
    sigma_m: float      # Gaussian error of the measurement [Msun]
    factor: float       # B^spec / B^mass of that star
    bayes: float        # B^spec = B^mass * factor
    bcat_coeff: float   # coeff in B_cat = 1 + coeff * p_det


def spectroscopic_forecast(
    res: MassCatalogResult,
    frac: float = c.SPEC_MASS_FRAC,
    m_spec: float | None = None,
    name: str = "S301",
) -> SpectroscopicForecast:
    """Forecast for a spectroscopic mass m_spec of star ``name``, measured
    with fractional precision ``frac`` (Gaussian, sigma_m = frac * m_spec), on
    top of the mass check ``res``. Default m_spec: the line mass a/C, i.e. a
    measurement lying on the energy line. The star keeps its K magnitude; the
    other stars keep their B_i^mass."""
    if not frac > 0.0:
        raise ValueError("frac must be positive")
    star = res[name]
    m_spec = star.m_line if m_spec is None else float(m_spec)
    if not m_spec > 0.0:
        raise ValueError("m_spec must be positive")
    sigma_m = frac * m_spec
    model, k, a_au = res.model, star.k_mag, star.star.a_au
    # densities of m_spec given K; the normalizations of the K densities cancel
    cond_pair = model.pair_density_spec(k, a_au, m_spec, sigma_m) / star.l_pair
    cond_null = model.null_density_spec(k, m_spec, sigma_m, star.k_lim_null) / star.p_null
    factor = cond_pair / cond_null
    bayes = star.bayes * factor
    total = sum(r.bayes for r in res.results if r.star.name != name) + bayes
    return SpectroscopicForecast(
        name=name, m_spec=m_spec, sigma_m=sigma_m, factor=factor, bayes=bayes,
        bcat_coeff=total / res.n_stars - 1.0,
    )
