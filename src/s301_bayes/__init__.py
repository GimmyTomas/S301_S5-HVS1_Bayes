"""Reproduction code for the statistical analysis of
"Is S301 the Captured Companion of the Hypervelocity Star S5-HVS1?"
(Caputo & Tomaselli).

Entry points: scripts/robustness.py (the paper's table), scripts/run_analysis.py
(per-star output), scripts/robustness_mass.py (mass check and forecast), or

    from s301_bayes import compute_catalog, compute_catalog_mass
    orbit = compute_catalog(1.1)        # CatalogResult, orbit only (fiducial)
    mass = compute_catalog_mass(1.1)    # MassCatalogResult, with K magnitudes
"""

from . import constants
from .bayes import compute_all, compute_catalog
from .catalog import load_catalog, load_s301
from .mass import compute_catalog_mass, spectroscopic_forecast

__version__ = "1.1.0"

__all__ = [
    "constants",
    "compute_all",
    "compute_catalog",
    "compute_catalog_mass",
    "spectroscopic_forecast",
    "load_catalog",
    "load_s301",
]
