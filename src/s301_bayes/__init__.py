"""Reproduction code for the statistical analysis of
"Is S301 the Captured Companion of the Hypervelocity Star S5-HVS1?"
(Caputo & Tomaselli).

Entry point: scripts/run_analysis.py, or

    from s301_bayes.bayes import compute_all
    tables = compute_all()
"""

from . import constants
from .bayes import compute_all, compute_catalog
from .catalog import load_catalog, load_s301

__all__ = ["constants", "compute_all", "compute_catalog", "load_catalog", "load_s301"]
