"""Load the S-star catalog used in the analysis.

Orbital elements come from Gillessen et al. 2017 (ApJ 837, 30, Table 3) for
most stars, overridden by the updated four-star fit of GRAVITY Collaboration
2024 (A&A 692, A242, Table B.1) for S2, S29, S38 and S55, plus S301 from
GRAVITY Collaboration 2026 (arXiv:2607.12664). Angular semi-major axes are
converted to AU with the distance R0 fitted by the respective paper.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from . import constants as c


@dataclass(frozen=True)
class Star:
    name: str
    a_au: float        # semi-major axis [AU]
    a_err_au: float
    e: float           # eccentricity
    e_err: float
    source: str        # 'Gillessen+2017', 'GRAVITY+2024' or 'GRAVITY+2026'


def data_dir() -> Path:
    env = os.environ.get("S301_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    return rows


def load_all_stars() -> list[Star]:
    """All bound S-stars, with GRAVITY 2024 values overriding Gillessen 2017."""
    d = data_dir()

    gravity = {
        r["star"]: Star(
            name=r["star"],
            a_au=float(r["a_arcsec"]) * c.R0_GRAVITY24_PC,
            a_err_au=float(r["a_err_arcsec"]) * c.R0_GRAVITY24_PC,
            e=float(r["e"]),
            e_err=float(r["e_err"]),
            source="GRAVITY+2024",
        )
        for r in _read_csv(d / "gravity2024_tableB1.csv")
    }

    stars = []
    for r in _read_csv(d / "gillessen2017_table3.csv"):
        if float(r["e"]) >= 1.0 or float(r["a_arcsec"]) <= 0.0:
            continue  # S111 is formally unbound
        if r["star"] in gravity:
            stars.append(gravity[r["star"]])
        else:
            stars.append(
                Star(
                    name=r["star"],
                    a_au=float(r["a_arcsec"]) * c.R0_GILLESSEN17_PC,
                    a_err_au=float(r["a_err_arcsec"]) * c.R0_GILLESSEN17_PC,
                    e=float(r["e"]),
                    e_err=float(r["e_err"]),
                    source="Gillessen+2017",
                )
            )
    return stars


def load_s301(solution: str = "primary") -> Star:
    for r in _read_csv(data_dir() / "s301.csv"):
        if r["solution"] == solution:
            return Star(
                name="S301",
                a_au=float(r["a_AU"]),
                a_err_au=float(r["a_err_AU"]),
                e=float(r["e"]),
                e_err=float(r["e_err"]),
                source="GRAVITY+2026",
            )
    raise ValueError(f"unknown S301 solution {solution!r}")


def load_catalog(a_range: tuple[float, float] = c.A_SUPPORT) -> list[Star]:
    """The N stars with semi-major axis inside the support of f_pair
    (S301 included), sorted by semi-major axis."""
    stars = [s for s in load_all_stars() if a_range[0] <= s.a_au <= a_range[1]]
    stars.append(load_s301("primary"))
    return sorted(stars, key=lambda s: s.a_au)
