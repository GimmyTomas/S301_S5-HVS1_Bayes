"""Bayes factors: per star and catalog-level.

B_i = f_pair(a_i, e_i) / f_0(a_i, e_i), and the catalog-level factor

    B_cat = (1 - p_det) + (p_det / N) * sum_i B_i  =  1 + coeff * p_det,
    coeff = (1/N) * sum_i B_i - 1,

where p_det is the probability that the companion survived and is detected in
the catalog of the N stars with semi-major axis inside the support. The null
eccentricity exponent alpha (default: thermal, alpha = 1) enters f_0 only.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import constants as c
from .catalog import Star, load_catalog, load_s301
from .likelihoods import PairLikelihood, null_likelihood


@dataclass(frozen=True)
class StarResult:
    star: Star
    f_pair: float
    f_null: float

    @property
    def bayes(self) -> float:
        return self.f_pair / self.f_null


@dataclass(frozen=True)
class CatalogResult:
    gamma: float
    results: list[StarResult]
    mirror_b301: float  # B for S301's mirrored orbital solution
    alpha: float = c.ALPHA_THERMAL  # null eccentricity exponent, p(e) ~ e^alpha

    @property
    def n_stars(self) -> int:
        return len(self.results)

    @property
    def bcat_coeff(self) -> float:
        """coeff in B_cat = 1 + coeff * p_det."""
        return sum(r.bayes for r in self.results) / self.n_stars - 1.0

    def __getitem__(self, name: str) -> StarResult:
        for r in self.results:
            if r.star.name == name:
                return r
        raise KeyError(name)


def compute_catalog(
    gamma, pair: PairLikelihood | None = None, alpha: float = c.ALPHA_THERMAL
) -> CatalogResult:
    """Bayes factors of the catalog for density slope gamma and null
    eccentricity exponent alpha. f_pair does not depend on alpha, so one
    PairLikelihood(gamma) can be passed in and reused for every alpha."""
    pair = pair or PairLikelihood(gamma)
    if pair.gamma != gamma:
        raise ValueError(f"pair likelihood has gamma = {pair.gamma}, not {gamma}")
    results = [
        StarResult(
            star=s,
            f_pair=pair(s.a_au, s.e),
            f_null=null_likelihood(s.a_au, s.e, gamma, alpha),
        )
        for s in load_catalog()
    ]
    mirror = load_s301("mirror")
    mirror_b301 = pair(mirror.a_au, mirror.e) / null_likelihood(
        mirror.a_au, mirror.e, gamma, alpha
    )
    return CatalogResult(
        gamma=gamma, results=results, mirror_b301=mirror_b301, alpha=alpha
    )


def compute_all(gammas=c.GAMMAS, alpha=c.ALPHA_THERMAL) -> dict[float, CatalogResult]:
    return {g: compute_catalog(g, alpha=alpha) for g in gammas}
