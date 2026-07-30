from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

from stellarator_diagnostics.boozer import infer_boozer_resolution
from stellarator_diagnostics.plots import (
    plot_boozer_surface_files,
    plot_boozer_surfaces,
)
from stellarator_diagnostics.qi import compute_goodman_qi, diagnose_goodman_qi
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


def make_qi_boozmn(path: Path, perturbation=0.0):
    ns = 9
    modes = [0, 0] if perturbation == 0 else [0, 0, 1]
    toroidal_modes = [0, 4] if perturbation == 0 else [0, 4, 0]
    coefficients = [2.0, 0.2] if perturbation == 0 else [2.0, 0.2, perturbation]
    iota = np.full(ns, 0.65)
    xr.Dataset(
        {
            "nfp_b": xr.DataArray(4),
            "ns_b": xr.DataArray(ns),
            "jlist": ("packed_radius", [5]),
            # Real boozmn files commonly retain full-radius phi_b and iota_b
            # while bmnc_b contains only the packed jlist surfaces.
            "phi_b": ("full_radius", np.linspace(0, 1, ns)),
            "iota_b": ("full_radius", iota),
            "ixm_b": ("mn", modes),
            "ixn_b": ("mn", toroidal_modes),
            "bmnc_b": (("packed_radius", "mn"), [coefficients]),
        }
    ).to_netcdf(path)


def test_boozer_jlist_uses_packed_half_grid_and_full_iota(tmp_path):
    source = tmp_path / "boozmn_qi.nc"
    make_qi_boozmn(source)
    adapter = BoozerAdapter(source)
    try:
        assert np.allclose(adapter.available_surfaces(), [(5 - 1.5) / 8])
        assert np.array_equal(adapter.surface_labels(), [5])
        assert adapter.iota_at(adapter.available_surfaces()[0]) == 0.65
    finally:
        adapter.close()


def test_goodman_qi_exact_field_has_zero_residual(tmp_path):
    source = tmp_path / "boozmn_exact_qi.nc"
    make_qi_boozmn(source)
    result = compute_goodman_qi(source, nalpha=16, nphi=65, nlevels=65)
    assert len(result.data) == 1
    assert result.data.loc[0, "f_QI"] < 1e-24
    assert result.data.loc[0, "squash_residual"] < 1e-24


def test_goodman_qi_detects_alpha_dependent_wells_and_writes_outputs(tmp_path):
    source = tmp_path / "boozmn_non_qi.nc"
    make_qi_boozmn(source, perturbation=0.02)
    result, outputs = diagnose_goodman_qi(
        source,
        tmp_path / "qi",
        nalpha=20,
        nphi=65,
        nlevels=65,
    )
    assert result.data.loc[0, "f_QI"] > 1e-8
    assert {path.name for path in outputs} >= {
        "goodman_qi_residual.csv",
        "goodman_qi_summary.json",
        "goodman_qi_residual.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
