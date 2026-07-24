"""Catalog construction: bound orbits, GRAVITY 2024 overrides, selection cut.
Anchored to the values quoted in the paper."""

import numpy as np

from s301_bayes.catalog import load_all_stars, load_catalog, load_s301


def test_39_bound_orbits():
    assert len(load_all_stars()) == 39  # 40 orbits minus the unbound S111


def test_catalog_has_18_stars():
    cat = load_catalog()
    assert len(cat) == 18
    assert {s.name for s in cat} == {
        "S2", "S4", "S8", "S9", "S12", "S13", "S14", "S17", "S18", "S21",
        "S23", "S29", "S38", "S39", "S55", "S60", "S175", "S301",
    }


def test_gravity2024_overrides():
    cat = {s.name: s for s in load_catalog()}
    for name in ("S2", "S29", "S38", "S55"):
        assert cat[name].source == "GRAVITY+2024"
    # S29 as quoted in the paper: a = 3230 +- 15 AU, e = 0.9688
    assert abs(cat["S29"].a_au - 3230.0) < 5.0
    assert np.isclose(cat["S29"].e, 0.9688, atol=1e-4)


def test_gillessen_anchors():
    cat = {s.name: s for s in load_catalog()}
    # values quoted in the paper (a converted with R0 = 8.32 kpc)
    assert abs(cat["S14"].a_au - 2380.0) < 5.0 and np.isclose(cat["S14"].e, 0.9761)
    assert abs(cat["S175"].a_au - 3450.0) < 10.0 and np.isclose(cat["S175"].e, 0.9867)
    assert cat["S14"].source == "Gillessen+2017"


def test_only_three_bound_orbits_above_e093():
    high = sorted(s.name for s in load_all_stars() if s.e > 0.93)
    assert high == ["S14", "S175", "S29"]


def test_s301():
    s = load_s301()
    assert (s.a_au, s.e) == (687.0, 0.9832)
    m = load_s301("mirror")
    assert m.e == 0.9824
