import numpy as np
import pytest

from molecular_simulations.analysis.sasa import SASA

def test_get_sphere_points_on_unit_sphere():
    s = SASA.get_sphere.__func__(n_points=256)
    assert s.shape == (256, 3)
    radii = np.linalg.norm(s, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)
