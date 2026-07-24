#!/usr/bin/env python3
"""Reproduce the auxiliary numbers quoted in the paper around the statistical
analysis (everything except the Bayes-factor table, which is produced by
scripts/run_analysis.py).

Usage:  python scripts/paper_numbers.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes.catalog import load_all_stars, load_s301
from s301_bayes.evolution import evolve_star


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
    print(f"\nPredicted S301 mass from Eq. (1):")
    print(
        f"  m_b = {m_req:.3f} Msun  ('~1.37')   "
        f"[{m_req * (1 - rel):.3f}, {m_req * (1 + rel):.3f}] with a and m_e errors"
    )
    print(
        f"  S301 pericenter: r_p = a(1-e) = {s301.a_au * (1 - s301.e):.1f} AU "
        f"('~11.5 AU');  j_301 = sqrt(1-e^2) = {c.J_301:.4f}  ('~0.18')"
    )

    print(f"\nAngular-momentum spread over the flight time, Delta j = j_301 sqrt(t_f/t_j):")
    for tj in (2e7, 3e7):
        print(f"  t_j = {tj:.0e} yr:  Delta j = {c.J_301 * np.sqrt(c.T_FLIGHT / tj):.4f}")
    print("  ('~0.07-0.09')")

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

    n_cat = sum(
        1 for s in stars.values() if c.A_SUPPORT[0] <= s.a_au <= c.A_SUPPORT[1]
    ) + 1  # + S301
    print(
        f"\nCatalog: N = {n_cat} stars with a in "
        f"[{c.A_SUPPORT[0]:.0f}, {c.A_SUPPORT[1]:.0f}] AU "
        "('N = 18', interval quoted rounded as [550, 3500] AU)"
    )


if __name__ == "__main__":
    main()
