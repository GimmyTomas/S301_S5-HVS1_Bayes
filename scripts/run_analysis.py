#!/usr/bin/env python3
"""Compute every Bayes factor of the paper's statistical analysis.

Deterministic (Fokker-Planck based): repeated runs give identical numbers.
Runtime: a couple of minutes on a laptop.

Usage:  python scripts/run_analysis.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes.bayes import compute_catalog
from s301_bayes.likelihoods import PairLikelihood

HEADLINE = ["S301", "S14", "S29", "S175"]


def main():
    print("S301 / S5-HVS1 catalog-level Bayes-factor analysis")
    print(f"  line coefficient : {c.LINE_COEFF_AU:.2f} AU/Msun")
    print(
        f"  a support        : [{c.A_SUPPORT[0]:.1f}, {c.A_SUPPORT[1]:.1f}] AU"
        "  (paper quotes the rounded [550, 3500] AU)"
    )
    print(f"  absorbing radius : {c.R_ABSORB_AU} AU (pericenter)")

    tables = {}
    for gamma in c.GAMMAS:
        t0 = time.time()
        pair = PairLikelihood(gamma)
        tables[gamma] = compute_catalog(gamma, pair=pair)
        print(f"\ngamma = {gamma}  (computed in {time.time() - t0:.0f} s)")
        print(f"  {'star':<6}{'a [AU]':>9}{'e':>9}{'P_surv':>9}{'B_i':>12}  source")
        for r in tables[gamma].results:
            s = r.star
            psurv = pair.p_surv(s.a_au)
            print(
                f"  {s.name:<6}{s.a_au:>9.1f}{s.e:>9.4f}{float(psurv):>9.3f}"
                f"{r.bayes:>12.4g}  {s.source}"
            )
        n = tables[gamma].n_stars
        print(f"  N = {n} stars;  B_cat = 1 + {tables[gamma].bcat_coeff:.3g} * p_det")
        print(
            f"  S301 mirrored solution (e = 0.9824): "
            f"B_301 = {tables[gamma].mirror_b301:.4g}"
        )

    print("\n" + "=" * 58)
    print("Paper table (Bayes factors):")
    print(f"  {'':<12}" + "".join(f"gamma = {g:<8}" for g in c.GAMMAS))
    for name in HEADLINE:
        row = f"  B_{name[1:]:<10}"
        for g in c.GAMMAS:
            row += f"{tables[g][name].bayes:<16.3g}"
        print(row)
    row = f"  {'B_cat':<12}"
    for g in c.GAMMAS:
        row += f"1 + {tables[g].bcat_coeff:.3g} p_det  "
    print(row)


if __name__ == "__main__":
    main()
