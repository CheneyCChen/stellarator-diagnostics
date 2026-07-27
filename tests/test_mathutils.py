import numpy as np

from stellarator_diagnostics.mathutils import (
    edge_extrapolate,
    fourier_cos,
    fourier_sin,
    interp_radial,
)


def test_fourier_reconstruction():
    theta, zeta = np.meshgrid([0.2, 0.7], [0.1, 0.4], indexing="ij")
    xm = np.array([0, 1])
    xn = np.array([0, 4])
    coeff = np.array([2.0, 0.5])
    expected = 2 + 0.5 * np.cos(theta - 4 * zeta)
    assert np.allclose(fourier_cos(coeff, xm, xn, theta, zeta), expected)
    expected_sin = 0.5 * np.sin(theta - 4 * zeta)
    assert np.allclose(fourier_sin([0, 0.5], xm, xn, theta, zeta), expected_sin)


def test_radial_interpolation():
    values = np.array([[0, 2], [1, 4], [2, 6]], dtype=float)
    assert np.allclose(interp_radial(values, 0.25), [0.5, 3])
    half_values = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert np.allclose(interp_radial(half_values, 0.25, half=True, full_size=5), [1.5])


def test_edge_extrapolate():
    assert np.allclose(edge_extrapolate([1, 2, 3]), (0.5, 3.5))
