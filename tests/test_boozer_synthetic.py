from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

from stellarator_diagnostics.boozer import infer_boozer_resolution
from stellarator_diagnostics.plots import (
    plot_boozer_surface_files,
    plot_boozer_surfaces,
)
from stellarator_diagnostics.vmec import BoozerAdapter

matplotlib.use("Agg")


def make_boozmn(path: Path):
    s = np.array([0.25, 0.5, 0.75, 1.0])
    bmnc = np.zeros((len(s), 3))
    bmnc[:, 0] = 2.0
    bmnc[:, 1] = np.linspace(0.02, 0.12, len(s))
    bmnc[:, 2] = np.linspace(0.01, 0.05, len(s))
    xr.Dataset(
        {
            "nfp_b": xr.DataArray(4),
            "s_b": ("radius", s),
            "ixm_b": ("mn", [0, 1, 2]),
            "ixn_b": ("mn", [0, 4, 8]),
            "bmnc_b": (("radius", "mn"), bmnc),
        }
    ).to_netcdf(path)


def test_boozer_multiple_surface_line_plot(tmp_path):
    source = tmp_path / "boozmn_test.nc"
    output = tmp_path / "boozer_surfaces.png"
    make_boozmn(source)
    adapter = BoozerAdapter(source)
    assert np.allclose(adapter.available_surfaces(), [0.25, 0.5, 0.75, 1.0])
    field = adapter.field_map(s=0.5, ntheta=24, nzeta=20)
    assert field.values.shape == (24, 20)
    plot_boozer_surfaces(source, output)
    assert output.stat().st_size > 10_000
    outputs = plot_boozer_surface_files(source, tmp_path / "separate", surfaces=[0.25, 0.75])
    assert len(outputs) == 2
    assert all(path.stat().st_size > 10_000 for path in outputs)


def test_infer_boozer_resolution_uses_vmec_nyquist_modes(tmp_path):
    source = tmp_path / "wout_resolution.nc"
    xr.Dataset(
        {
            "nfp": xr.DataArray(4),
            "mpol": xr.DataArray(9),
            "ntor": xr.DataArray(9),
            "xm": ("mn", [0, 8]),
            "xn": ("mn", [0, 36]),
            "xm_nyq": ("mn_nyq", [0, 8, 16]),
            # VMEC xn already contains NFP, hence 64 / 4 = 16.
            "xn_nyq": ("mn_nyq", [-64, 0, 64]),
        }
    ).to_netcdf(source)
    assert infer_boozer_resolution(source) == (16, 16)
