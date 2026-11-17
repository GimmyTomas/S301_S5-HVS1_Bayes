#!/usr/bin/env python3
"""Regenerate data/gillessen2017_table3.csv from VizieR (catalog J/ApJ/837/30).

Not needed to reproduce the paper numbers (the CSV is version-controlled);
provided so the provenance of the data file can be independently verified.

Usage:  python scripts/fetch_gillessen_table.py
"""

import sys
import urllib.request
from pathlib import Path

QUERY = "/viz-bin/asu-tsv?-source=J/ApJ/837/30/table3&-out=Star,a,e_a,e,e_e&-out.max=100"
# VizieR at CDS (Strasbourg), then its CfA mirror
HOSTS = ("https://vizier.cds.unistra.fr", "https://vizier.cfa.harvard.edu")
OUT = Path(__file__).resolve().parents[1] / "data" / "gillessen2017_table3.csv"

HEADER = """\
# Orbital parameters of the 40 stars with determined orbits in the Galactic Center.
# Source: Gillessen et al. 2017, ApJ 837, 30 (arXiv:1611.09144), Table 3,
# transcribed from the machine-readable version at VizieR, catalog J/ApJ/837/30/table3
# (https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/837/30; retrieved {date}).
# Semi-major axes are angular (arcsec); the fit of that paper has
# R0 = 8.32 kpc, so 1 arcsec = 8320 AU (conversion applied in code, not here).
# S111 is formally unbound (e > 1) and is excluded by the bound-orbit selection.
# Columns: star, a_arcsec, a_err_arcsec, e, e_err
star,a_arcsec,a_err_arcsec,e,e_err
"""


def main():
    text = None
    for host in HOSTS:
        try:
            with urllib.request.urlopen(host + QUERY, timeout=60) as r:
                text = r.read().decode()
            break
        except OSError as err:
            print(f"{host}: {err}", file=sys.stderr)
    if text is None:
        sys.exit("could not reach VizieR")
    rows = []
    past_header = False
    for line in text.splitlines():
        if line.startswith("---"):
            past_header = True
            continue
        if not past_header or not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        star, a, ea, e, ee = parts[0], *map(float, parts[1:5])
        rows.append(f"{star},{a:.4f},{ea:.4f},{e:.4f},{ee:.4f}")
    if len(rows) != 40:
        sys.exit(f"expected 40 rows, got {len(rows)}; VizieR output changed?")
    from datetime import date

    OUT.write_text(HEADER.format(date=date.today().isoformat()) + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
