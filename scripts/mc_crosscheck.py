#!/usr/bin/env python3
"""Monte Carlo validation of the deterministic (Fokker-Planck) pipeline.

Runs the forward model as a literal seeded random-walk simulation and compares
against the quadrature+Fokker-Planck numbers used for the paper:

 1. per-star check at fixed semi-major axis: survival probability and the
    final eccentricity density at the star's measured eccentricity;
 2. global check of the full joint density f_pair(a, e): brute-force binary
    sampling, line mapping, capture eccentricity and SDE evolution, with
    local window counting around each headline star.

Usage:  python scripts/mc_crosscheck.py [--n-star 4000000] [--n-global 20000000]
                                        [--steps 1200] [--seed 301] [--gamma 1.1]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s301_bayes import constants as c
from s301_bayes.binary_sampling import sample_companions
from s301_bayes.catalog import load_catalog
from s301_bayes.evolution import d_jj, evolve_mc, evolve_star, j_loss_cone
from s301_bayes.initial_conditions import sample_j0
from s301_bayes.likelihoods import PairLikelihood

HEADLINE = ["S301", "S14", "S29", "S175"]


def per_star_check(stars, gamma, n, steps, rng):
    print(f"\n--- per-star conditional check (gamma={gamma}, n={n:.0e}) ---")
    print(
        f"  {'star':<6}{'P_surv FP':>11}{'P_surv MC':>11}{'dev':>8}"
        f"{'p_e FP':>10}{'p_e MC':>10}{'dev':>8}"
    )
    for s in stars:
        sol = evolve_star(s.a_au, gamma)
        j0 = sample_j0(n, s.a_au, rng)
        jf, alive = evolve_mc(
            j0, d_jj(s.a_au, gamma), c.T_FLIGHT, steps, rng, j_loss_cone(s.a_au)
        )
        ps_mc = alive.mean()
        ps_err = np.sqrt(ps_mc * (1 - ps_mc) / n)
        ji = np.sqrt(1.0 - s.e**2)
        delta = 0.01
        cnt = int(np.sum(np.abs(jf[alive] - ji) < delta))
        pj_mc = cnt / (alive.sum() * 2 * delta)          # p(j | survived)
        pe_mc = pj_mc * s.e / ji                          # convert to e-density
        pe_mc_err = pe_mc / max(np.sqrt(cnt), 1.0)
        pe_fp = sol["p_e"](s.e)
        print(
            f"  {s.name:<6}{sol['p_surv']:>11.4f}{ps_mc:>11.4f}"
            f"{(ps_mc / sol['p_surv'] - 1) * 100:>7.2f}%"
            f"{pe_fp:>10.4f}{pe_mc:>10.4f}{(pe_mc / pe_fp - 1) * 100:>7.2f}%"
            f"   (MC stat err: {ps_err / ps_mc * 100:.2f}%, {pe_mc_err / pe_mc * 100:.2f}%)"
        )


def global_check(stars, gamma, n, steps, rng, pair):
    print(f"\n--- global joint-density check (gamma={gamma}, n={n:.0e}) ---")
    t0 = time.time()
    mb, me = sample_companions(n, rng)
    a = mb * c.LINE_COEFF_AU
    kappa = np.exp(rng.normal(np.log(c.KAPPA_MEDIAN), c.KAPPA_SIGMA_LN, n))
    x = np.minimum(kappa * ((mb + me) / c.M_BH) ** (1.0 / 3.0), 1.0)
    j0 = np.sqrt(x * (2.0 - x))
    jf, alive = evolve_mc(
        j0, d_jj(a, gamma), c.T_FLIGHT, steps, rng, j_loss_cone(a)
    )
    print(f"  simulated in {time.time() - t0:.0f} s")
    print(f"  overall survival: MC {alive.mean():.4f}  vs  Z = {pair.Z:.4f}")

    a_s, j_s = a[alive], jf[alive]
    print(f"  {'star':<6}{'f_pair FP':>12}{'f_pair MC':>12}{'dev':>8}{'stat err':>10}")
    for s in stars:
        ji = np.sqrt(1.0 - s.e**2)
        da, dj = 25.0, 0.01
        cnt = int(np.sum((np.abs(a_s - s.a_au) < da) & (np.abs(j_s - ji) < dj)))
        f_aj = cnt / (alive.sum() * 4 * da * dj)          # density in (a, j) | survived
        f_mc = f_aj * s.e / ji                            # -> density in (a, e)
        f_fp = pair(s.a_au, s.e)
        stat = 100.0 / max(np.sqrt(cnt), 1.0)
        print(
            f"  {s.name:<6}{f_fp:>12.5g}{f_mc:>12.5g}"
            f"{(f_mc / f_fp - 1) * 100:>7.1f}%{stat:>9.1f}%"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-star", type=int, default=4_000_000)
    ap.add_argument("--n-global", type=int, default=20_000_000)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--gamma", type=float, default=None, help="default: both")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    catalog = {s.name: s for s in load_catalog()}
    stars = [catalog[n] for n in HEADLINE]
    gammas = [args.gamma] if args.gamma is not None else list(c.GAMMAS)

    for gamma in gammas:
        pair = PairLikelihood(gamma)
        per_star_check(stars, gamma, args.n_star, args.steps, rng)
        global_check(stars, gamma, args.n_global, args.steps, rng, pair)


if __name__ == "__main__":
    main()
