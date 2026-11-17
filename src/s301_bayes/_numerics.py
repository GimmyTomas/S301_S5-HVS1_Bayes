"""Small numerical helpers shared by the package."""

import numpy as np

# Trapezoidal rule: numpy.trapezoid (NumPy >= 2.0), formerly numpy.trapz
# (removed in NumPy 2.4). The two are the same function.
trapezoid = getattr(np, "trapezoid", None) or np.trapz
