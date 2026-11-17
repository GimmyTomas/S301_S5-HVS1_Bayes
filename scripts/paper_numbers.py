#!/usr/bin/env python3
"""Reproduce the auxiliary numbers quoted in the paper around the statistical
analysis (everything except the table of Bayes factors, which is produced by
scripts/robustness.py), each next to the value quoted in the text.

Usage:  python scripts/paper_numbers.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes import mass as M
from s301_bayes.catalog import load_all_stars, load_catalog, load_s301
from s301_bayes.evolution import d_jj, evolve_star
from s301_bayes.likelihoods import PairLikelihood, fit_alpha


def main():
    C = c.LINE_COEFF_AU
    print("Mass-energy line, Eq. (1):")
    print(f"  a_b = {C:.1f} AU * (m_b/Msun)   ('~500 AU' in the paper)")
    print(
        f"  intrinsic uncertainty from m_e = 2.35 +- {c.M_EJ_SIGMA}: "
        f"{c.M_EJ_SIGMA / c.M_EJ * 100:.1f}%  ('approximately 3%')"
    )

    s301 = load_s301()
    m_req = s301.a_au / C
    # m_b = a * m_e * v_inf^2 / (G M_BH)  ->  relative errors of a and m_e add
    rel = np.hypot(s301.a_err_au / s301.a_au, c.M_EJ_SIGMA / c.M_EJ)
    print("\nPredicted S301 mass from Eq. (1):")
    print(
        f"  m_b = {m_req:.3f} Msun  ('~1.37')   "
        f"[{m_req * (1 - rel):.3f}, {m_req * (1 + rel):.3f}] with a and m_e errors"
    )
    print(
        f"  S301 pericenter: r_p = a(1-e) = {s301.a_au * (1 - s301.e):.1f} AU "
        f"('~11.5 AU');  j_301 = sqrt(1-e^2) = {c.J_301:.4f}  ('~0.18')"
    )

    t_e = 1e9  # yr, energy relaxation time adopted in the paper (lower limit)
    print(
        f"\nSemi-major-axis diffusion over the flight time, sqrt(t_f/t_E) with t_E = 1 Gyr: "
        f"{np.sqrt(c.T_FLIGHT / t_e) * 100:.1f}%  ('~7%')"
    )

    print("\nAngular-momentum spread over the flight time, Delta j = j_301 sqrt(t_f/t_j):")
    for tj in (2e7, 3e7):
        print(f"  t_j = {tj:.0e} yr:  Delta j = {c.J_301 * np.sqrt(c.T_FLIGHT / tj):.4f}")
    print("  ('~0.07-0.09')")
    print(
        f"  1/D_jj = t_j/j_301^2 = {1.0 / d_jj(c.A_301, c.GAMMA_FIDUCIAL):.1e} yr "
        f"at a = {c.A_301:.0f} AU (t_j = {c.T_J_301:.0e} yr)  ('~9e8 yr')"
    )

    print("\nCapture eccentricity scale (m_bin/M_BH)^(1/3):")
    for mbin in (2.2 + 1.1, 2.5 + 7.0):
        print(f"  m_bin = {mbin:.1f} Msun:  1-e ~ {(mbin / c.M_BH) ** (1 / 3):.4f}")
    print("  ('~10^-2', Eq. (3))")

    sol = evolve_star(s301.a_au, gamma=1.1)
    j, p = sol["j"], sol["p_final"]
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    e_med = np.sqrt(1 - np.interp(0.5, cdf, j) ** 2)
    e_lo = np.sqrt(1 - np.interp(0.95, cdf, j) ** 2)  # 95th pct in j = 5th in e
    print(
        f"\nEvolved eccentricity at a = {s301.a_au:.0f} AU (gamma=1.1, survivors): "
        f"median e = {e_med:.3f}, 5th percentile e = {e_lo:.3f}  ('e >~ 0.95 today')"
    )

    stars = {s.name: s for s in load_all_stars()}
    print(f"\nBound orbits in the input catalogs: {len(stars)}  ('39 known bound orbits')")
    high_e = [s for s in stars.values() if s.e > 0.93]
    print(f"  stars with e > 0.93: {', '.join(s.name for s in high_e)}")
    print("\nLine-required masses (a_i / line coefficient) vs paper quotes 4.8, 6.5, 7:")
    for name in ("S14", "S29", "S175"):
        s = stars[name]
        print(
            f"  {name:<5} a = {s.a_au:6.0f} +- {s.a_err_au:3.0f} AU, e = {s.e:.4f} "
            f"({s.source}):  m_line = {s.a_au / C:.2f} Msun"
        )

    phot = M.load_photometry()
    rel = M.mass_magnitude_relation("mamajek")
    a_lo, a_hi = c.A_K_RANGE
    print(
        f"\nPhotometric masses for A_K = {a_lo:g}-{a_hi:g} (Mamajek relation), and how much "
        "fainter the star\nis than its line mass requires; paper quotes 7.5-8.7, 4.5-5.1, "
        "2.9-3.4, S301 1.3-1.5 Msun;\nS175 1.4-1.7 mag and S29 0.5-0.8 mag fainter:"
    )
    for name in ("S14", "S29", "S175", "S301"):
        k = phot[name].k_mag
        a_au = s301.a_au if name == "S301" else stars[name].a_au
        dk = [k - rel.k_mag(a_au / C, a_k) for a_k in (a_hi, a_lo)]
        print(
            f"  {name:<5} m_K = {k:5.2f}:  m_hat = {M.photometric_mass(k, a_lo):.2f}-"
            f"{M.photometric_mass(k, a_hi):.2f} Msun;  K - K(a/C) = {dk[0]:+.2f} to {dk[1]:+.2f} mag"
        )

    n_cat = sum(
        1 for s in stars.values() if c.A_SUPPORT[0] <= s.a_au <= c.A_SUPPORT[1]
    ) + 1  # + S301
    print(
        f"\nCatalog: N = {n_cat} stars with a in "
        f"[{c.A_SUPPORT[0]:.0f}, {c.A_SUPPORT[1]:.0f}] AU "
        "('N = 18', interval quoted rounded as [550, 3500] AU)"
    )

    others = [s.e for s in load_catalog() if s.name != "S301"]
    alpha, sigma = fit_alpha(others)
    print(
        f"\nNull eccentricity exponent, p(e) ~ e^alpha, fit to the {len(others)} stars "
        "other than S301:"
    )
    print(
        f"  alpha = {alpha:.2f} +- {sigma:.2f} (maximum likelihood, Fisher error)  "
        "('1.7 +- 0.7')"
    )

    # Mass check, fiducial settings (scripts/robustness_mass.py)
    pair = PairLikelihood(c.GAMMA_FIDUCIAL)
    res = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair)
    print("\nMass check (scripts/robustness_mass.py):")
    budget = np.sqrt(0.3**2 + 0.15**2 + 0.2**2)  # S301 photometry, A_K, relation
    print(
        f"  sigma_K budget: sqrt(0.3^2 + 0.15^2 + 0.2^2) = {budget:.2f} mag "
        f"-> sigma_K = {c.SIGMA_K:g} mag"
    )
    print(
        f"  photometric masses (Mamajek, A_K = {c.A_K_GC:g}), "
        "vs paper quotes 1.38, 8.1, 4.8, 3.1:"
    )
    for name in ("S301", "S14", "S29", "S175"):
        r = res[name]
        dk = r.k_mag - r.k_line
        side = "brighter" if dk < 0 else "fainter"
        print(
            f"  {name:<5} K = {r.k_mag:5.2f}:  m_hat = {r.m_hat:4.2f} Msun;  "
            f"line K(a/C) = {r.k_line:5.2f}, observed {abs(dk):.2f} mag {side}"
        )
    print("  ('0.02 mag' for S301, '1.5 mag fainter' for S175)")
    k_mist = M.MassModel(pair_relation="mist").k_pair(res["S301"].m_line)
    print(
        f"  MIST pair relation: S301 K(a/C) = {k_mist:.2f}, observed "
        f"{k_mist - res['S301'].k_mag:.2f} mag brighter  ('0.36 mag brighter')"
    )
    print(
        "  mean detection probability of the surviving companions: "
        f"Z_s/Z = {1.0 / res.z_ratio:.3f}  ('0.98')"
    )
    f = M.spectroscopic_forecast(res)
    print(
        f"  outlook: S301 with a spectroscopic mass on the line, {c.SPEC_MASS_FRAC:.0%} precision "
        f"(m_spec = {f.m_spec:.2f} +- {f.sigma_m:.2f} Msun):\n"
        f"    B_301 = {f.bayes:.0f}, B_cat = 1 + {f.bcat_coeff:.1f} p_det"
    )
    lo, hi = c.M_NULL_RANGE
    r12 = M.compute_catalog_mass(c.GAMMA_FIDUCIAL, pair, m_null=(lo, 12.0))
    print(
        f"  null IMF on [{lo:g}, 12] instead of [{lo:g}, {hi:g}] Msun: "
        f"B_301 {(r12['S301'].bayes / res['S301'].bayes - 1) * 100:+.1f}%, "
        f"B_cat coeff {(r12.bcat_coeff / res.bcat_coeff - 1) * 100:+.1f}%  ('-2.4%')"
    )


if __name__ == "__main__":
    main()
