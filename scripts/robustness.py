#!/usr/bin/env python3
"""The paper's table: the fiducial Bayes factors and their robustness checks.

Columns: the fiducial model (gamma = 1.1, thermal null p(e) = 2e), then one
ingredient changed at a time: the Bahcall-Wolf slope gamma = 1.75 (thermal
null); power-law null eccentricities p(e) = (1+alpha) e^alpha at the fiducial
gamma (constants.ALPHAS_ROBUSTNESS: 1.7, the maximum-likelihood fit to the
catalog stars other than S301, and 2.6, Gillessen et al. 2009; overridable
with --alpha); and each star's observed K magnitude as a further observable
("with mass", s301_bayes.mass; details and sensitivity in
scripts/robustness_mass.py). The alpha and mass columns reuse the fiducial
PairLikelihood (only f_0 depends on alpha): one Fokker-Planck setup per gamma.

Usage:  python scripts/robustness.py [--alpha 1.7 2.6]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes.bayes import compute_catalog
from s301_bayes.catalog import load_catalog
from s301_bayes.likelihoods import PairLikelihood, fit_alpha
from s301_bayes.mass import compute_catalog_mass

HEADLINE = ["S301", "S14", "S29", "S175"]

# (row label, formatter of a CatalogResult column)
ROWS = [(f"B_{n[1:]}", lambda r, n=n: f"{r[n].bayes:.3g}") for n in HEADLINE] + [
    ("B_cat coeff", lambda r: f"{r.bcat_coeff:.3g}"),
    ("N", lambda r: f"{r.n_stars}"),
    ("B_301 mirror", lambda r: f"{r.mirror_b301:.3g}"),
]


def robustness_columns(alphas=c.ALPHAS_ROBUSTNESS):
    """[(label, CatalogResult)]: fiducial, other gammas, alphas, with mass."""
    g0 = c.GAMMA_FIDUCIAL
    pairs = {g: PairLikelihood(g) for g in {g0, *c.GAMMAS}}
    cols = [("fiducial", compute_catalog(g0, pairs[g0]))]
    cols += [(f"gamma={g}", compute_catalog(g, pairs[g])) for g in c.GAMMAS if g != g0]
    cols += [(f"alpha={a:g}", compute_catalog(g0, pairs[g0], alpha=a)) for a in alphas]
    cols.append(("with mass", compute_catalog_mass(g0, pairs[g0])))
    return cols


def format_table(columns):
    """Plain-text table of ROWS for a list of (label, CatalogResult) columns."""
    w0 = max(len(name) for name, _ in ROWS) + 2
    widths = [max(10, len(label) + 2) for label, _ in columns]
    lines = [" " * w0 + "".join(f"{lab:>{w}}" for (lab, _), w in zip(columns, widths))]
    for name, fmt in ROWS:
        cells = "".join(f"{fmt(r):>{w}}" for (_, r), w in zip(columns, widths))
        lines.append(f"{name:<{w0}}{cells}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robustness table of the Bayes factors.")
    ap.add_argument(
        "--alpha", type=float, nargs="+", default=c.ALPHAS_ROBUSTNESS, metavar="A",
        help="null eccentricity exponents, p(e) = (1+A) e^A, at the fiducial gamma "
             "(default: %(default)s)",
    )
    args = ap.parse_args(argv)

    others = [s.e for s in load_catalog() if s.name != "S301"]
    alpha, sigma = fit_alpha(others)
    print("S301 / S5-HVS1 Bayes factors: robustness checks")
    print(f"  fiducial : gamma = {c.GAMMA_FIDUCIAL}, thermal null p(e) = 2e (alpha = 1)")
    print("  others   : one ingredient changed; null p(e) = (1+alpha) e^alpha")
    print(
        f"  alpha    : maximum-likelihood fit to the {len(others)} catalog stars other "
        f"than S301: alpha = {alpha:.2f} +- {sigma:.2f};"
    )
    print(
        "             Gillessen et al. 2009 (early-type stars outside the clockwise disk): "
        "2.6 +- 0.9"
    )
    print("  with mass: fiducial x (Z/Z_s) L_pair(K|a) / p_0(K|det), observed K magnitudes,")
    print(
        f"             sigma_K = {c.SIGMA_K:g} mag, detection K <= {c.K_LIM:g} "
        "(scripts/robustness_mass.py)"
    )
    print("  B_cat = 1 + coeff * p_det;  B_301 mirror: S301 mirrored solution\n")
    print(format_table(robustness_columns(args.alpha)))


if __name__ == "__main__":
    main()
