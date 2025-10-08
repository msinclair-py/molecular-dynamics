import numpy as np
import pytest

from molecular_simulations.analysis.fingerprinter import (
    dist_mat, electrostatic, electrostatic_sum,
    lennard_jones, lennard_jones_sum, unravel_index
)

def test_unravel_index_shapes(x, y):
    i, j = unravel_index(5, 7)
    assert i.shape == j.shape
    assert i.size == 5 * 7
    assert i.max() == 4 and j.max() == 6

def test_dist_mat_basic():
    xyz1 = np.array([0., 0., 0.],
                    [1., 0., 0.])
    xyz2 = np.array([0., 1., 0.])
    D = dist_mat(xyz1, xyz2)
    assert D.shape == (2, 1)
    assert np.isclose(D[0, 0], 1.)
    assert np.isclose(D[1, 0], np.sqrt(2.))

def test_electrostatic_symmetry_and_sign():
    d = 0.5  # distance
    e1, e2 =  0.5, -0.5
    e3, e4 = -0.5,  0.5
    e_ab = electrostatic(d, e1, e2)
    e_ba = electrostatic(d, e2, e1)
    e_cd = electrostatic(d, e3, e4)
    # Symmetric in swapping particles
    assert np.isclose(e_ab, e_ba)
    # Opposite-sign charges should yield negative energy; same magnitude pair should match
    assert e_ab < 0
    assert np.isclose(e_ab, e_cd)

def test_electrostatic_sum_vectorization_matches_scalar():
    rng = np.random.default_rng(0)
    n, m = 4, 3
    D = rng.random((n, m)) + 0.2
    qi = rng.uniform(-1, 1, size=n)
    qj = rng.uniform(-1, 1, size=m)

    # Scalar accumulation
    scalar = 0.0
    for a in range(n):
        for b in range(m):
            scalar += electrostatic(D[a, b], qi[a], qj[b])

    # Vectorized helper
    vec = electrostatic_sum(D, qi, qj)
    assert np.isclose(vec, scalar)

def test_lj_basic_properties():
    # Attractive well around sigma, repulsive at very small r
    e_far = lennard_jones(5.0, 1.0, 1.0, 0.2, 0.2)
    e_mid = lennard_jones(1.5, 1.0, 1.0, 0.2, 0.2)
    e_close = lennard_jones(0.5, 1.0, 1.0, 0.2, 0.2)
    assert e_far > e_mid  # tends toward 0 from below/above depending on combination
    assert e_close > e_mid  # repulsive wall at short distance

def test_lj_sum_matches_manual_sum():
    rng = np.random.default_rng(1)
    n, m = 3, 2
    D = rng.random((n, m)) + 0.3
    si = rng.random(n) + 0.5
    sj = rng.random(m) + 0.5
    ei = rng.random(n) * 0.3 + 0.05
    ej = rng.random(m) * 0.3 + 0.05

    manual = 0.0
    for a in range(n):
        for b in range(m):
            manual += lennard_jones(D[a, b], si[a], sj[b], ei[a], ej[b])

    summed = lennard_jones_sum(D, si, sj, ei, ej)
    assert np.isclose(summed, manual)
