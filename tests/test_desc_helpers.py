import numpy as np

from stellarator_diagnostics.desc_reader import _rho_derivative_to_s


def test_desc_volume_derivative_is_converted_from_rho_to_s():
    rho = np.linspace(0, 1, 6)
    s = rho**2
    # V=rho**2 has dV/drho=2*rho and dV/ds=1, including its axis limit.
    converted = _rho_derivative_to_s(s, 2 * rho)
    assert np.allclose(converted, 1.0)
