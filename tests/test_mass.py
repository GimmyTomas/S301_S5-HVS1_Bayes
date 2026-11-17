"""Mass robustness check in K-magnitude space: the mass-magnitude relations,
the K densities under the two hypotheses, the detection normalizations, and
B_i^mass = B_i (Z / Z_s) L_pair(K_i | a_i) / p_0(K_i | det)."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from s301_bayes._numerics import trapezoid
from s301_bayes import constants as c
from s301_bayes import mass as M
from s301_bayes.bayes import compute_catalog
from s301_bayes.catalog import load_all_stars, load_catalog
from s301_bayes.likelihoods import PairLikelihood, u_a

C = c.LINE_COEFF_AU
HEADLINE = ["S301", "S14", "S29", "S175"]


@pytest.fixture(scope="module")
def pair_fiducial():
    return PairLikelihood(c.GAMMA_FIDUCIAL)


@pytest.fixture(scope="module")
def res_mass(pair_fiducial):
    return M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial)


def _sig(x, n=3):
    """x rounded to n significant figures."""
    return float(f"{x:.{n}g}")


def _quad_m(f, lo, hi, rel):
    """int_lo^hi f(m) dm by adaptive quadrature, split at the relation's nodes."""
    pts = [m for m in 10.0**rel.log_m if lo < m < hi]
    val, _ = quad(f, lo, hi, points=pts or None, limit=2000, epsabs=0.0, epsrel=1e-12)
    return val


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Gillessen et al. 2017, Table 3 (m_K), and S301 (GRAVITY+2026)
TABLE3_MK = {
    "S2": 13.95, "S4": 14.4, "S8": 14.5, "S9": 15.1, "S12": 15.5, "S13": 15.8,
    "S14": 15.7, "S17": 15.3, "S18": 16.7, "S21": 16.9, "S23": 17.8, "S29": 16.7,
    "S38": 17.0, "S39": 16.8, "S55": 17.5, "S60": 16.3, "S175": 17.5, "S301": 19.3,
}


def test_photometry_covers_catalog():
    ph = M.load_photometry()
    assert {s.name for s in load_all_stars()} | {"S301"} == set(ph)  # 39 bound + S301
    for s in load_catalog():
        assert ph[s.name].k_mag == TABLE3_MK[s.name]
    late = {n for n, p in ph.items() if n in TABLE3_MK and p.sptype == "l"}
    assert late == {"S17", "S21", "S38"}
    with pytest.raises(TypeError):  # the cached mapping is shared: read-only
        ph["S301"] = ph["S2"]


def test_constants():
    assert c.R0_PHOT_PC == 8178.0
    assert c.A_K_GC == 2.55 and c.A_K_RANGE == (2.4, 2.7)
    # error budget: photometry 0.3, extinction 0.15, relation scatter 0.2 mag
    assert c.SIGMA_K == 0.4 and round(math.sqrt(0.3**2 + 0.15**2 + 0.2**2), 2) == 0.39
    assert c.K_LIM == 20.0
    assert c.LINE_SIGMA_FRAC == 0.03
    assert c.M_NULL_RANGE == (0.5, 20.0)
    assert np.isclose(M.distance_modulus(), 14.5632, atol=1e-4)


# ---------------------------------------------------------------------------
# Mass-magnitude relations
# ---------------------------------------------------------------------------


def test_mamajek_relation():
    rel = M.mass_magnitude_relation("mamajek")
    assert len(rel.log_m) == 80 and np.all(np.diff(rel.log_m) > 0)
    assert rel.mass_range == pytest.approx((0.075, 20.2), rel=1e-14)
    # through every tabulated row (O9V ... L2V), linear in log10 m in between
    assert np.allclose(rel.abs_mag_k(10.0**rel.log_m), rel.abs_mag, rtol=0, atol=1e-13)
    m1, m2 = 1.33, 1.38  # F5V, F4V
    mid = math.sqrt(m1 * m2)
    assert rel.abs_mag_k(mid) == pytest.approx(0.5 * (2.291 + 2.188), abs=1e-13)
    k_sun = 3.236 + M.distance_modulus() + 2.55  # G2V, 1 Msun
    assert rel.k_mag(1.0, a_k=2.55) == pytest.approx(k_sun, abs=1e-12)


def test_mist_relation():
    rel = M.mass_magnitude_relation("mist")
    assert len(rel.log_m) == 274 and np.all(np.diff(rel.log_m) > 0)
    assert rel.mass_range == pytest.approx((0.52292, 6.91512), rel=1e-14)
    assert np.allclose(rel.abs_mag_k(10.0**rel.log_m), rel.abs_mag, rtol=0, atol=1e-13)
    # clamped beyond the end of the isochrone (pair masses reach 7 Msun)
    assert rel.abs_mag_k(7.0) == rel.abs_mag_k(9.0) == rel.abs_mag[-1] == -2.3991


def test_unknown_relation():
    with pytest.raises(ValueError):
        M.mass_magnitude_relation("parsec")


def test_relations_monotone_where_used():
    """M_Ks decreases with mass over every mass that enters an integral, except
    on short stretches whose excursion is far below sigma_K: Mamajek's A3V ->
    A5V segment (1.86 -> 1.88 Msun) and the main-sequence hook at the end of
    the MIST isochrone (above 6.88 Msun)."""
    lo, hi = c.M_NULL_RANGE  # contains the pair masses, line width included
    s = 8.0 * c.LINE_SIGMA_FRAC
    assert lo < c.MB_MIN * (1.0 - s) and c.MB_MAX * (1.0 + s) < hi

    mam = M.mass_magnitude_relation("mamajek")
    inside = (mam.log_m >= math.log10(lo)) & (mam.log_m <= math.log10(hi))
    m, mk = 10.0 ** mam.log_m[inside], mam.abs_mag[inside]
    up = np.flatnonzero(np.diff(mk) >= 0.0)
    assert len(up) == 1 and (m[up[0]], m[up[0] + 1]) == pytest.approx((1.86, 1.88), rel=1e-12)
    assert mk[up[0] + 1] - mk[up[0]] == pytest.approx(0.16, abs=1e-12)
    assert 0.16 < 0.5 * c.SIGMA_K

    mist = M.mass_magnitude_relation("mist")
    m, mk = 10.0**mist.log_m, mist.abs_mag
    i_min = int(np.argmin(mk))
    assert m[i_min] == pytest.approx(6.8847, abs=1e-12)
    assert np.all(np.diff(mk[: i_min + 1]) < 0.0)
    assert np.max(mk[i_min:]) - mk[i_min] < 0.5 * c.SIGMA_K   # 0.135 mag


def test_inverse_relation_consistent():
    """The display inverse (log10 M linear in M_Ks) undoes K(m) exactly, except
    on the non-monotonic A2V-A6V stretch (1.83-1.98 Msun)."""
    rel = M.mass_magnitude_relation("mamajek")
    m = np.geomspace(*c.M_NULL_RANGE, 2000)
    m = m[(m <= 1.83) | (m >= 1.98)]
    assert np.allclose(M.photometric_mass(rel.k_mag(m)), m, rtol=1e-12, atol=0)
    for s in load_catalog():
        k = M.load_photometry()[s.name].k_mag
        assert rel.k_mag(M.photometric_mass(k)) == pytest.approx(k, abs=1e-12)


@pytest.mark.parametrize(
    "name, m_lo, m_mid, m_hi",
    [
        ("S301", 1.31, 1.38, 1.46),
        ("S14", 7.53, 8.07, 8.66),
        ("S29", 4.45, 4.78, 5.12),
        ("S175", 2.88, 3.14, 3.41),
    ],
)
def test_photometric_masses(name, m_lo, m_mid, m_hi, res_mass):
    """A_K = 2.4 / 2.55 / 2.7 (numbers quoted in the paper text)."""
    k = M.load_photometry()[name].k_mag
    got = [M.photometric_mass(k, a_k) for a_k in (2.4, c.A_K_GC, 2.7)]
    assert [round(x, 2) for x in got] == [m_lo, m_mid, m_hi]
    assert res_mass[name].m_hat == got[1]


# ---------------------------------------------------------------------------
# Densities of K
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m_range", [c.M_NULL_RANGE, (0.5, 12.0), (0.3, 20.0)])
def test_imf_normalized(m_range):
    val, _ = quad(M.imf_pdf, *m_range, args=(m_range,), epsabs=0, epsrel=1e-12)
    assert abs(val - 1.0) < 1e-10
    lo, hi = m_range
    assert M.imf_pdf(0.99 * lo, m_range) == M.imf_pdf(1.01 * hi, m_range) == 0.0


@pytest.mark.parametrize("relation", ["mamajek", "mist"])
@pytest.mark.parametrize(
    "a_au, k",
    [
        (687.0, 19.3),          # S301
        (2382.0, 15.7),         # S14
        (3444.0, 17.5),         # S175
        (c.A_SUPPORT[0], 20.0),
        (c.A_SUPPORT[1], 15.0),
    ],
)
def test_pair_density_matches_quad(relation, a_au, k):
    model = M.MassModel(pair_relation=relation)
    m0, s = a_au / C, c.LINE_SIGMA_FRAC * a_au / C
    f = lambda m: norm.pdf(m, m0, s) * norm.pdf(k, model.k_pair(m), model.sigma_k)
    ref = _quad_m(f, m0 - 10.0 * s, m0 + 10.0 * s, model.pair_rel)
    assert model.pair_density(k, a_au) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("relation", ["mamajek", "mist"])
@pytest.mark.parametrize("line_sigma", [0.0, c.LINE_SIGMA_FRAC, 0.07])
@pytest.mark.parametrize("a_au", [c.A_SUPPORT[0], 687.0, 2382.0, c.A_SUPPORT[1]])
def test_pair_density_normalized(relation, line_sigma, a_au):
    """L_pair(K | a) integrates to 1 over all K."""
    model = M.MassModel(line_sigma=line_sigma, pair_relation=relation)
    k0 = model.k_pair(a_au / C)
    val, _ = quad(lambda k: model.pair_density(k, a_au), k0 - 10.0, k0 + 10.0,
                  limit=500, epsabs=0, epsrel=1e-12)
    assert abs(val - 1.0) < 1e-10


@pytest.mark.parametrize("a_au", [c.A_SUPPORT[0], 687.0, 2382.0])
def test_pair_detection_point_mass(a_au):
    """With no line width, P_det(a/C) is exactly the detected part of L_pair."""
    model = M.MassModel(line_sigma=0.0)
    val, _ = quad(lambda k: model.pair_density(k, a_au), model.k_lim - 12.0, model.k_lim,
                  epsabs=0, epsrel=1e-12)
    assert model.pair_detection(a_au) == pytest.approx(val, rel=1e-10)


@pytest.mark.parametrize("sigma_k", [0.3, c.SIGMA_K])
@pytest.mark.parametrize("k", [19.3, 17.5, 15.7, 13.95, 20.0])
def test_null_density_matches_quad(sigma_k, k):
    model = M.MassModel(sigma_k=sigma_k)
    rel, (lo, hi) = model.null_rel, c.M_NULL_RANGE
    num = _quad_m(lambda m: M.imf_pdf(m) * norm.pdf(k, model.k_null(m), sigma_k), lo, hi, rel)
    det = _quad_m(
        lambda m: M.imf_pdf(m) * norm.cdf((c.K_LIM - model.k_null(m)) / sigma_k), lo, hi, rel
    )
    assert model.null_detection() == pytest.approx(det, rel=1e-10)
    assert model.null_density(k) == pytest.approx(num / det, rel=1e-10)


@pytest.mark.parametrize(
    "kw, k_lim",
    [
        ({}, None),
        (dict(sigma_k=0.3, k_lim=19.7), None),
        (dict(sigma_k=0.5, m_null=(0.5, 12.0)), None),
        ({}, 18.0),  # per-star limit of the null normalization
    ],
)
def test_null_density_normalized_over_detected(kw, k_lim):
    """p_0(K | det) integrates to 1 over K <= K_lim and vanishes above."""
    model = M.MassModel(**kw)
    lim = model.k_lim if k_lim is None else k_lim
    val, _ = quad(lambda k: model.null_density(k, k_lim), 8.0, lim,
                  limit=500, epsabs=0, epsrel=1e-12)
    assert abs(val - 1.0) < 1e-10
    assert model.null_density(lim + 1e-9, k_lim) == 0.0


def test_selection_ratio(pair_fiducial):
    """Z / Z_s against adaptive quadrature over a, and its limits."""
    model = M.MassModel()
    zr = model.selection_ratio(pair_fiducial)
    w = lambda a: float(u_a(a)) * float(pair_fiducial.p_surv(a))
    pts = [C * m for m in 10.0**model.pair_rel.log_m if c.MB_MIN < m < c.MB_MAX]
    kw = dict(points=pts, limit=1000, epsabs=0, epsrel=1e-10)
    z, _ = quad(w, *c.A_SUPPORT, **kw)
    z_s, _ = quad(lambda a: w(a) * norm.cdf((c.K_LIM - model.k_pair(a / C)) / c.SIGMA_K),
                  *c.A_SUPPORT, **kw)
    assert zr == pytest.approx(z / z_s, rel=1e-6)
    assert _sig(zr, 4) == 1.025
    # a faint limit detects every companion; the line width does not enter Z_s
    assert M.MassModel(k_lim=30.0).selection_ratio(pair_fiducial) == 1.0
    assert M.MassModel(line_sigma=0.0).selection_ratio(pair_fiducial) == zr


def test_line_width_in_detection_is_negligible(pair_fiducial):
    """Averaging P_det over the line width (the exact detected fraction of
    L_pair) would change Z / Z_s by a relative 5e-4 only (module docstring)."""
    model = M.MassModel()
    a = np.linspace(*c.A_SUPPORT, 401)
    w = u_a(a) * pair_fiducial.p_surv(a)

    def p_det_exact(a_au):
        m0, s = a_au / C, c.LINE_SIGMA_FRAC * a_au / C
        f = lambda m: (
            math.exp(-0.5 * ((m - m0) / s) ** 2) / (s * math.sqrt(2.0 * math.pi))
            * ndtr((c.K_LIM - model.k_pair(m)) / c.SIGMA_K)
        )
        return _quad_m(f, m0 - 10.0 * s, m0 + 10.0 * s, model.pair_rel)

    exact = trapezoid(w, a) / trapezoid(w * [p_det_exact(x) for x in a], a)
    approx = trapezoid(w, a) / trapezoid(w * model.pair_detection(a), a)
    assert exact / approx - 1.0 == pytest.approx(5e-4, abs=1e-4)


# ---------------------------------------------------------------------------
# Catalog with K magnitudes
# ---------------------------------------------------------------------------


def test_bayes_mass_over_bayes_is_factor(res_mass):
    model = res_mass.model
    assert res_mass.n_stars == 18
    for r in res_mass.results:
        k = M.load_photometry()[r.star.name].k_mag
        assert r.k_mag == k and r.k_lim_null == c.K_LIM
        assert r.l_pair == model.pair_density(k, r.star.a_au)
        assert r.p_null == model.null_density(k)
        assert r.k_line == model.k_pair(r.star.a_au / C) and r.m_line == r.star.a_au / C
        assert r.z_ratio == res_mass.z_ratio
        assert r.factor == pytest.approx(res_mass.z_ratio * r.l_pair / r.p_null, rel=1e-15)
        assert r.bayes / r.bayes_orbital == pytest.approx(r.factor, rel=1e-14)
        assert r.f_pair / r.f_null == pytest.approx(r.bayes, rel=1e-12)
    # the mirrored solution of S301 has the same a, hence the same factor
    assert res_mass.mirror_b301 / res_mass.orbital.mirror_b301 == pytest.approx(
        res_mass["S301"].factor, rel=1e-14
    )
    mean = sum(r.bayes for r in res_mass.results) / 18
    assert res_mass.bcat_coeff == pytest.approx(mean - 1.0, rel=1e-14)


def test_fiducial_orbit_numbers_unchanged(pair_fiducial, res_mass):
    ref = compute_catalog(c.GAMMA_FIDUCIAL, pair_fiducial)
    assert f"{ref['S301'].bayes:.3g}" == "90.2"
    assert f"{ref.bcat_coeff:.3g}" == "4.23"
    assert [r.bayes for r in res_mass.orbital.results] == [r.bayes for r in ref.results]
    assert res_mass.orbital.mirror_b301 == ref.mirror_b301


def test_fiducial_numbers(res_mass):
    """Headline numbers of the mass check (regression values)."""
    assert _sig(res_mass.z_ratio, 4) == 1.025
    assert _sig(res_mass["S301"].factor) == 2.75
    assert [_sig(res_mass[n].bayes) for n in HEADLINE[:3]] == [248, 1.81, 1.70]
    assert _sig(res_mass["S175"].bayes, 2) == 0.0028
    assert _sig(res_mass.bcat_coeff, 4) == 12.99


# (settings, B_301, B_14, B_29, B_cat coefficient, Z/Z_s), 3 significant figures
VARIANTS = [
    (dict(sigma_k=0.3), 316, 0.235, 0.953, 16.6, 1.02),
    (dict(sigma_k=0.5), 207, 4.41, 2.01, 10.9, 1.03),
    (dict(k_lim=19.7), 226, 1.65, 1.55, 11.8, 1.06),
    (dict(k_lim=20.5), 293, 2.14, 2.01, 15.6, 1.00),
    (dict(a_k=2.4), 249, 4.03, 0.839, 13.1, 1.02),
    (dict(a_k=2.7), 219, 0.714, 2.98, 11.4, 1.04),
    (dict(line_sigma=0.0), 253, 1.69, 1.68, 13.3, 1.02),
    (dict(line_sigma=0.07), 225, 2.38, 1.82, 11.8, 1.02),
    (dict(pair_relation="mist"), 172, 2.26, 0.0359, 8.70, 1.06),
]


@pytest.mark.parametrize("kw, b301, b14, b29, coeff, z_ratio", VARIANTS)
def test_variant_numbers(pair_fiducial, kw, b301, b14, b29, coeff, z_ratio):
    """One setting changed at a time (regression values)."""
    r = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial, **kw)
    got = [r["S301"].bayes, r["S14"].bayes, r["S29"].bayes, r.bcat_coeff, r.z_ratio]
    assert [_sig(x) for x in got] == [b301, b14, b29, coeff, z_ratio]


def test_point_mass_limit(pair_fiducial, res_mass):
    """line_sigma = 0 gives the point-mass formula, factor =
    (Z/Z_s) N(K; K(a/C), sigma_K) / p_0(K | det), and small widths converge
    to it at second order."""
    r0 = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial, line_sigma=0.0)
    model = r0.model
    assert r0.z_ratio == res_mass.z_ratio
    for r in r0.results:
        point = norm.pdf(r.k_mag, model.k_pair(r.m_line), c.SIGMA_K)
        expected = r0.z_ratio * point / model.null_density(r.k_mag)
        assert r.factor == pytest.approx(expected, rel=1e-13)
    for s in (1e-3, 1e-4, 1e-5):
        narrow = M.MassModel(line_sigma=s)
        for name in HEADLINE:
            r = r0[name]
            assert narrow.pair_density(r.k_mag, r.star.a_au) == pytest.approx(
                r.l_pair, rel=300.0 * s**2
            )


def test_null_mass_range_insensitive(pair_fiducial, res_mass):
    lo, hi = c.M_NULL_RANGE
    # stars below 0.5 Msun are > 7 sigma_K fainter than K_lim: no effect at all
    r_lo = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial, m_null=(0.3, hi))
    for a, b in zip(r_lo.results, res_mass.results):
        assert a.bayes == pytest.approx(b.bayes, rel=1e-10)
    # dropping the stars above 12 Msun (2% of the detected null) changes the
    # relevant numbers by less than 3% (at most 2.4%)
    r_hi = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial, m_null=(lo, 12.0))
    for name in HEADLINE:
        assert r_hi[name].bayes == pytest.approx(res_mass[name].bayes, rel=0.03)
    assert r_hi.bcat_coeff == pytest.approx(res_mass.bcat_coeff, rel=0.03)


def test_per_star_null_limit(pair_fiducial, res_mass):
    """k_lim_star changes the null normalization of the named stars only."""
    g = c.GAMMA_FIDUCIAL
    ao = {s.name: 18.0 for s in load_catalog() if s.name != "S301"}
    r = M.compute_catalog_mass(g, pair_fiducial, k_lim_star=ao)
    assert r.z_ratio == res_mass.z_ratio
    assert r["S301"].bayes == res_mass["S301"].bayes
    assert r.mirror_b301 == res_mass.mirror_b301
    for a, b in zip(r.results, res_mass.results):
        assert a.l_pair == b.l_pair
        if a.star.name != "S301":
            assert a.k_lim_null == 18.0
            assert a.p_null == r.model.null_density(a.k_mag, 18.0) > b.p_null
    with pytest.raises(KeyError):
        M.compute_catalog_mass(g, pair_fiducial, k_lim_star={"S1": 18.0})
    with pytest.raises(ValueError):  # S23 has K = 17.8
        M.compute_catalog_mass(g, pair_fiducial, k_lim_star={"S23": 17.5})
    with pytest.raises(ValueError):  # S301 has K = 19.3
        M.compute_catalog_mass(g, pair_fiducial, k_lim=19.0)


@pytest.mark.parametrize(
    "kw",
    [
        dict(sigma_k=0.0),
        dict(line_sigma=-0.01),
        dict(line_sigma=0.2),
        dict(pair_relation="parsec"),
        dict(m_null=(0.05, 20.0)),
        dict(m_null=(0.5, 30.0)),
        dict(m_null=(5.0, 2.0)),
    ],
)
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        M.MassModel(**kw)


# ---------------------------------------------------------------------------
# Forecast: spectroscopic mass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relation", ["mamajek", "mist"])
@pytest.mark.parametrize("frac", [0.01, c.SPEC_MASS_FRAC, 0.10])
@pytest.mark.parametrize("m_spec", [1.2, 687.0 / C, 1.6])
def test_spec_densities_match_quad(relation, frac, m_spec):
    """Joint densities of (K, m_spec) against adaptive quadrature."""
    model = M.MassModel(pair_relation=relation)
    a_au, k, sm = 687.0, 19.3, frac * m_spec
    m0, s = a_au / C, model.line_sigma
    ref_pair = _quad_m(
        lambda m: norm.pdf(m, m0, s * m0) * norm.pdf(k, model.k_pair(m), model.sigma_k)
        * norm.pdf(m_spec, m, sm),
        m0 * (1 - 8 * s), m0 * (1 + 8 * s), model.pair_rel,
    )
    assert model.pair_density_spec(k, a_au, m_spec, sm) == pytest.approx(ref_pair, rel=1e-9)
    lo, hi = max(c.M_NULL_RANGE[0], m_spec - 12 * sm), min(c.M_NULL_RANGE[1], m_spec + 12 * sm)
    ref_null = _quad_m(
        lambda m: M.imf_pdf(m) * norm.pdf(k, model.k_null(m), model.sigma_k)
        * norm.pdf(m_spec, m, sm),
        lo, hi, model.null_rel,
    ) / model.null_detection()
    assert model.null_density_spec(k, m_spec, sm) == pytest.approx(ref_null, rel=1e-9)
    assert model.null_density_spec(model.k_lim + 0.01, m_spec, sm) == 0.0


def test_spec_densities_marginalize_to_k_densities():
    """Integrating the measured mass out returns the densities of K alone."""
    model = M.MassModel()
    a_au, k, sm = 687.0, 19.3, 0.07
    got, _ = quad(lambda ms: model.pair_density_spec(k, a_au, ms, sm), 0.6, 2.2,
                  epsabs=0.0, epsrel=1e-10, limit=200)
    assert got == pytest.approx(model.pair_density(k, a_au), rel=1e-8)
    # null: m_spec range wide enough for the stars that can have K = 19.3
    got, _ = quad(lambda ms: model.null_density_spec(k, ms, sm), 0.05, 6.0,
                  epsabs=0.0, epsrel=1e-10, limit=400)
    assert got == pytest.approx(model.null_density(k), rel=1e-6)


def test_spectroscopic_forecast_numbers(pair_fiducial, res_mass):
    """Outlook of the paper: S301 with a 5% mass measurement on the line."""
    f = M.spectroscopic_forecast(res_mass)
    assert f.name == "S301" and f.m_spec == res_mass["S301"].m_line
    assert f.sigma_m == c.SPEC_MASS_FRAC * f.m_spec
    assert f.bayes == res_mass["S301"].bayes * f.factor
    assert [_sig(f.factor), _sig(f.bayes), _sig(f.bcat_coeff)] == [2.63, 651, 35.4]
    # the other stars are untouched
    others = sum(r.bayes for r in res_mass.results if r.star.name != "S301")
    assert f.bcat_coeff == pytest.approx((others + f.bayes) / 18 - 1.0, rel=1e-14)
    # better precision helps on the line; the line width caps the gain
    coeffs = [M.spectroscopic_forecast(res_mass, x).bcat_coeff for x in (0.10, 0.05, 0.03, 0.01)]
    assert coeffs == sorted(coeffs)
    assert [_sig(x) for x in coeffs] == [22.8, 35.4, 47.3, 63.1]


@pytest.mark.parametrize("m_spec, coeff", [(1.1, -0.736), (1.2, 2.69), (1.55, 5.48), (1.7, -0.616)])
def test_spectroscopic_forecast_off_the_line(res_mass, m_spec, coeff):
    """A 5% measurement away from the line removes the evidence: B_cat < 1
    for m_spec <= 1.1 or >= 1.7 Msun."""
    f = M.spectroscopic_forecast(res_mass, m_spec=m_spec)
    assert _sig(f.bcat_coeff) == coeff
    assert f.bayes < res_mass["S301"].bayes


def test_spectroscopic_forecast_uninformative_limit(res_mass):
    """A measurement much coarser than the photometric mass changes nothing."""
    f = M.spectroscopic_forecast(res_mass, frac=30.0)
    assert f.factor == pytest.approx(1.0, abs=2e-3)


def test_spectroscopic_forecast_independent_of_null_limit(pair_fiducial, res_mass):
    """The normalization of the null K density cancels in the forecast factor."""
    r = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair_fiducial, k_lim_star={"S301": 19.5})
    f0, f1 = M.spectroscopic_forecast(res_mass), M.spectroscopic_forecast(r)
    assert f1.factor == pytest.approx(f0.factor, rel=1e-13)


def test_spectroscopic_forecast_invalid(res_mass):
    with pytest.raises(ValueError):
        M.spectroscopic_forecast(res_mass, frac=0.0)
    with pytest.raises(ValueError):
        M.spectroscopic_forecast(res_mass, m_spec=-1.0)
    with pytest.raises(KeyError):
        M.spectroscopic_forecast(res_mass, name="S0")
