#!/usr/bin/env python3
"""Mass robustness check: each star's observed K magnitude as an extra observable.

At the fiducial model (gamma = 1.1, thermal null), the orbital Bayes factors
are multiplied by the factor (see s301_bayes.mass)

    B_i^mass = B_i * (Z / Z_s) * L_pair(K_i | a_i) / p_0(K_i | det),

with K(m) = M_Ks(m) + DM + A_K from Mamajek's main-sequence table for both
hypotheses, Gaussian scatter sigma_K about K(m), and detection K <= K_lim.
Pair: true mass on the energy line, m ~ N(a/C, 0.03 a/C); the fiducial
companion-mass prior is conditioned on detection (Z / Z_s). Null: Kroupa IMF,
normalized over the detected stars.

Prints the per-star table, then a SENSITIVITY block (the orbit-only and the
fiducial with-mass rows, then one setting changed at a time), then a FORECAST
block: the Bayes factors if S301 also had a spectroscopic mass measurement
(s301_bayes.mass.spectroscopic_forecast), on and off the energy line.

Usage:  python scripts/robustness_mass.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes import mass as M
from s301_bayes.catalog import load_catalog
from s301_bayes.likelihoods import PairLikelihood

HEADLINE = ["S301", "S14", "S29", "S175"]

# Sensitivity settings, one changed at a time (fiducial values: constants.py)
SIGMA_KS = (0.3, 0.5)            # mag (fiducial SIGMA_K = 0.4)
K_LIMS = (19.7, 20.5)            # mag (fiducial K_LIM = 20)
A_KS = c.A_K_RANGE               # mag (fiducial A_K_GC = 2.55, the midpoint)
LINE_SIGMAS = (0.0, 0.07)        # fractional (fiducial LINE_SIGMA_FRAC = 0.03)
M_NULL_HI = 12.0                 # Msun, null IMF upper end (fiducial 20)
M_NULL_LO = 0.3                  # Msun, null IMF lower end (fiducial 0.5; m^-2.3
                                 # extended below 0.5)
# The 17 catalog stars other than S301 come from adaptive-optics imaging,
# complete only to K ~ 18: normalize their null with this brighter limit
K_LIM_AO = 18.0

# Forecast: spectroscopic mass of S301, m_spec +- frac * m_spec
SPEC_FRACS = (0.10, c.SPEC_MASS_FRAC, 0.03)             # on the line, m_spec = a/C
SPEC_MASSES = (1.1, 1.2, 1.3, 1.45, 1.55, 1.7)          # Msun, at SPEC_MASS_FRAC


def per_star_table(res):
    lines = [
        f"  {'star':<6}{'K':>6}{'Sp':>3}{'m_hat':>7}{'a/C':>7}{'K(a/C)':>8}"
        f"{'L_pair':>11}{'p_0(K|det)':>12}{'factor':>11}{'B_i':>11}{'B_i^mass':>11}"
    ]
    for r in res.results:
        lines.append(
            f"  {r.star.name:<6}{r.k_mag:>6.2f}{r.sptype:>3}{r.m_hat:>7.2f}{r.m_line:>7.2f}"
            f"{r.k_line:>8.2f}{r.l_pair:>11.3g}{r.p_null:>12.3g}{r.factor:>11.3g}"
            f"{r.bayes_orbital:>11.4g}{r.bayes:>11.4g}"
        )
    return "\n".join(lines)


def sensitivity_variants():
    """[(label, compute_catalog_mass keyword arguments)], fiducial excluded."""
    v = [(f"sigma_K = {s:g} mag", dict(sigma_k=s)) for s in SIGMA_KS]
    v += [(f"K_lim = {k:.1f}", dict(k_lim=k)) for k in K_LIMS]
    v += [(f"A_K = {a:.2f}", dict(a_k=a)) for a in A_KS]
    v += [(f"line width = {s:g}", dict(line_sigma=s)) for s in LINE_SIGMAS]
    v.append(
        ("pair relation: MIST, [Fe/H] = +0.25, log age = 7.7", dict(pair_relation="mist"))
    )
    lo, hi = c.M_NULL_RANGE
    v.append((f"null IMF on [{lo:g}, {M_NULL_HI:g}] Msun", dict(m_null=(lo, M_NULL_HI))))
    v.append((f"null IMF on [{M_NULL_LO:g}, {hi:g}] Msun", dict(m_null=(M_NULL_LO, hi))))
    ao = {s.name: K_LIM_AO for s in load_catalog() if s.name != "S301"}
    v.append(
        (
            f"null K_lim = {K_LIM_AO:g} for the {len(ao)} AO stars (S301: {c.K_LIM:g})",
            dict(k_lim_star=ao),
        )
    )
    return v


def main():
    g = c.GAMMA_FIDUCIAL
    pair = PairLikelihood(g)
    res = M.compute_catalog_mass(g, pair)
    orb = res.orbital
    lo, hi = c.M_NULL_RANGE

    print("S301 / S5-HVS1 Bayes factors with K magnitudes (mass robustness check)")
    print(f"  orbit    : gamma = {g}, thermal null p(e) = 2e (fiducial B_i)")
    print(
        f"  relation : K(m) = M_Ks(m) + DM + A_K, DM = 5 log10(R0/10 pc) = "
        f"{M.distance_modulus():.3f} (R0 = {c.R0_PHOT_PC:g} pc), A_K = {c.A_K_GC:g};"
    )
    print("             M_Ks linear in log10 m through Mamajek's dwarf table (v2022.04.16)")
    print(
        f"  scatter  : K ~ N(K(m), sigma_K), sigma_K = {c.SIGMA_K:g} mag;  "
        f"detection: K <= K_lim = {c.K_LIM:g}  (both hypotheses)"
    )
    print(
        f"  pair     : m ~ N(a/C, {c.LINE_SIGMA_FRAC:g} a/C), "
        f"C = {c.LINE_COEFF_AU:.2f} AU/Msun; L_pair(K|a) = int dm N(m) N(K; K(m), sigma_K)"
    )
    print(
        f"             prior m_b in [{c.MB_MIN:g}, {c.MB_MAX:g}] Msun, "
        f"conditioned on detection: Z/Z_s = {res.z_ratio:.4f}"
    )
    print("             Z_s = int da u(a) P_surv(a) Phi((K_lim - K(a/C))/sigma_K)")
    print(
        f"  null     : Kroupa xi(m) ~ m^-{c.IMF_ALPHA:g} on [{lo:g}, {hi:g}] Msun, "
        "normalized over the detected stars:"
    )
    print(
        "             p_0(K|det) = int dm xi N(K; K(m), sigma_K) "
        "/ int dm xi Phi((K_lim - K(m))/sigma_K)"
    )
    print("  factor   = (Z/Z_s) L_pair(K|a) / p_0(K|det);  B_i^mass = B_i * factor")
    print("  m_hat    : photometric mass (inverse relation), displayed only")
    print(
        "  [S17, S21, S38 are late-type giants: "
        "their main-sequence relation is only formal]\n"
    )
    print(per_star_table(res))
    print(
        f"\n  N = {res.n_stars};  B_cat = 1 + {orb.bcat_coeff:.3g} p_det (orbit only)"
        f"  ->  1 + {res.bcat_coeff:.4g} p_det (with mass)"
    )
    print(
        f"  S301 mirrored solution: B_301 = {orb.mirror_b301:.4g} (orbit only)"
        f"  ->  {res.mirror_b301:.4g} (with mass)"
    )

    w0 = 52
    hdr = "".join(f"{'B_' + n[1:]:>10}" for n in HEADLINE)
    hdr += f"{'B_cat coeff':>13}{'Z/Z_s':>8}"
    print("\n" + "=" * (2 + w0 + len(hdr)))
    print("SENSITIVITY (with mass; one setting changed at a time, others fiducial)")
    print(f"  {'':<{w0}}{hdr}")

    def row(label, r, z_ratio=None):
        cells = "".join(f"{r[n].bayes:>10.3g}" for n in HEADLINE)
        z = "-" if z_ratio is None else f"{z_ratio:.4f}"
        return f"  {label:<{w0}}{cells}{r.bcat_coeff:>13.4g}{z:>8}"

    print(row("orbit only (no mass)", orb))
    print(row("with mass, fiducial", res, res.z_ratio))
    for label, kw in sensitivity_variants():
        r = M.compute_catalog_mass(g, pair, **kw)
        print(row(label, r, r.z_ratio))

    s301 = res["S301"]
    print("\n" + "=" * (2 + w0 + len(hdr)))
    print("FORECAST: S301 with a spectroscopic mass m_spec +- sigma_m, in addition to K")
    print("  B_301^mass is multiplied by p_pair(m_spec | K, a) / p_0(m_spec | K);")
    print("  the other stars keep their B_i^mass")
    print(f"  {'m_spec [Msun]':<26}{'sigma_m/m_spec':>15}{'factor':>10}{'B_301':>10}{'B_cat coeff':>13}")

    def forecast_row(label, **kw):
        f = M.spectroscopic_forecast(res, **kw)
        return (
            f"  {label:<26}{f.sigma_m / f.m_spec:>15.0%}{f.factor:>10.3g}"
            f"{f.bayes:>10.3g}{f.bcat_coeff:>13.4g}"
        )

    print(f"  {'none (with mass, fiducial)':<26}{'-':>15}{1.0:>10.3g}{s301.bayes:>10.3g}"
          f"{res.bcat_coeff:>13.4g}")
    for frac in SPEC_FRACS:
        print(forecast_row(f"{s301.m_line:.2f} (on the line, a/C)", frac=frac))
    for m_spec in SPEC_MASSES:
        print(forecast_row(f"{m_spec:.2f}", m_spec=m_spec))


if __name__ == "__main__":
    main()
