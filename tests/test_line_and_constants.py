"""The mass-energy line and derived constants, anchored to the values quoted
in the paper."""

import numpy as np

from s301_bayes import constants as c


def test_line_coefficient():
    # a_b = G M_BH / (m_e v_inf^2) * m_b  ~  500 AU per solar mass
    assert 500.0 < c.LINE_COEFF_AU < 503.0


def test_predicted_s301_mass():
    # the paper (and its auto-generated numbers) quote 1.370 Msun at a = 687 AU
    assert abs(c.A_301 / c.LINE_COEFF_AU - 1.370) < 0.005


def test_support_interval():
    lo, hi = c.A_SUPPORT
    # quoted rounded as [550, 3500] AU in the paper
    assert abs(lo - 550.0) < 5.0
    assert abs(hi - 3500.0) < 15.0
    assert np.isclose(hi / lo, c.MB_MAX / c.MB_MIN)


def test_j301():
    assert np.isclose(c.J_301, np.sqrt(1.0 - c.E_301**2))
    assert abs(c.J_301 - 0.18) < 0.005
