import numpy as np

from stellarator_diagnostics.diagnostics import nfp_resonances, rational_surfaces


def test_rational_surface_crossing():
    result = rational_surfaces(
        np.linspace(0, 1, 11),
        np.linspace(0.68, 0.79, 11),
        max_denominator=4,
    )
    by_name = {item["rational"]: item for item in result}
    assert "3/4" in by_name
    assert np.isclose(by_name["3/4"]["s"], (0.75 - 0.68) / 0.11)
    assert "2/3" not in by_name


def test_nfp_compatible_resonances():
    result = nfp_resonances(0.68, 0.79, nfp=4)
    by_name = {item["rational"]: item for item in result}
    assert (by_name["8/11"]["m"], by_name["8/11"]["n"]) == (11, 8)
    assert (by_name["3/4"]["m"], by_name["3/4"]["n"]) == (16, 12)
    assert by_name["3/4"]["in_profile"]
    assert not by_name["4/5"]["in_profile"]
