from pathlib import Path

import numpy as np
import xarray as xr

from stellarator_diagnostics.readers import load_equilibrium


def make_wout(path: Path):
    ns = 5
    mn = 3
    mn_nyq = 2
    rmnc = np.zeros((ns, mn))
    rmnc[:, 0] = 1.5
    rmnc[:, 1] = np.linspace(0, 0.2, ns)
    zmns = np.zeros((ns, mn))
    zmns[:, 1] = np.linspace(0, 0.2, ns)
    bmnc = np.zeros((ns, mn_nyq))
    bmnc[:, 0] = 2.0
    bmnc[:, 1] = 0.1
    ds = xr.Dataset(
        {
            "ns": xr.DataArray(ns),
            "nfp": xr.DataArray(4),
            "mpol": xr.DataArray(3),
            "ntor": xr.DataArray(2),
            "aspect": xr.DataArray(7.5),
            "volume_p": xr.DataArray(12.0),
            "iotaf": ("rad", np.linspace(0.68, 0.78, ns)),
            "presf": ("rad", np.linspace(1e5, 0, ns)),
            "vp": ("rad", np.linspace(10, 8, ns)),
            "DMerc": ("rad", np.linspace(-0.1, 0.2, ns)),
            "xm": ("mn", [0, 1, 1]),
            "xn": ("mn", [0, 0, 4]),
            "xm_nyq": ("mn_nyq", [0, 1]),
            "xn_nyq": ("mn_nyq", [0, 4]),
            "rmnc": (("rad", "mn"), rmnc),
            "zmns": (("rad", "mn"), zmns),
            "bmnc": (("rad", "mn_nyq"), bmnc),
        }
    )
    ds.to_netcdf(path)


def test_vmec_reader_and_geometry(tmp_path):
    path = tmp_path / "wout_test.nc"
    make_wout(path)
    eq = load_equilibrium(path)
    assert eq.backend == "VMEC"
    assert eq.nfp == 4
    assert np.isclose(eq.scalars["iota_edge"], 0.78)
    assert np.isfinite(eq.scalars["magnetic_well"])
    surf = eq.surface(s=1, ntheta=16, nphi=12)
    assert surf.R.shape == (16, 12)
    section = eq.section(s=1, phi=np.pi / 8, ntheta=65)
    assert np.allclose(section.R[0], section.R[-1], atol=1e-13)
    assert np.allclose(section.Z[0], section.Z[-1], atol=1e-13)
    field = eq.field_map(s=0.5, ntheta=10, nzeta=8)
    assert field.values.shape == (10, 8)
    assert np.isclose(field.values.mean(), 2.0, atol=1e-12)
