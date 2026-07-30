from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from stellarator_diagnostics.external import (
    diagnose_cobra,
    diagnose_dkes,
    diagnose_neo,
    read_cobra,
    read_dkes,
    read_neo,
)
from stellarator_diagnostics.runners import (
    SolverDependencyError,
    SolverExecutionError,
    run_cobra_solver,
    run_neo_solver,
    write_cobra_input,
    write_neo_input,
)


def test_neo_basic_format_and_plot(tmp_path: Path):
    source = tmp_path / "neo_out.case"
    source.write_text(
        "2 1.0D-04 0.2 0.70 2.0 5.0\n"
        "3 8.0D-05 0.3 0.72 2.0 5.0\n"
        "4 6.0D-05 0.4 0.74 2.0 5.0\n",
        encoding="utf-8",
    )
    result = read_neo(source)
    assert result.eout_swi == 1
    assert np.isclose(result.data.loc[0, "epsilon_eff"], (1e-4) ** (2 / 3))
    diagnosed, outputs = diagnose_neo(source, tmp_path / "neo")
    assert diagnosed.summary()["surface_count"] == 3
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_dkes_bounds_and_plots(tmp_path: Path):
    source = tmp_path / "results.case"
    rows = ["DKES results", "19 numeric columns"]
    for efield in (0.0, 0.1):
        for cmul in (1e-4, 1e-3, 1e-2):
            base = 1.0 / (1.0 + 100 * cmul)
            values = [
                cmul,
                efield,
                0,
                0,
                0.95 * base,
                1.05 * base,
                -0.12 * base,
                -0.08 * base,
                0.45 * base,
                0.55 * base,
                1,
                1,
                1,
                1e-8,
                1,
                1,
                1,
                1,
                1,
            ]
            rows.append(" ".join(f"{value:.9e}" for value in values))
    source.write_text("\n".join(rows), encoding="utf-8")
    result = read_dkes(source)
    assert len(result.data) == 6
    assert np.isclose(result.data.iloc[0]["L11"], 1 / (1 + 100e-4))
    diagnosed, outputs = diagnose_dkes(source, tmp_path / "dkes")
    assert diagnosed.summary()["efield_count"] == 2
    assert len(outputs) == 2
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_cobra_new_and_legacy_formats(tmp_path: Path):
    modern = tmp_path / "cobra_grate.modern"
    modern.write_text(
        "0.0 0.0 3\n"
        "2 0.2 -0.10\n"
        "3 0.4 -0.20\n"
        "4 0.6 0.05\n"
        "1.57 0.0 3\n"
        "2 0.2 -0.08\n"
        "3 0.4 -0.22\n"
        "4 0.6 0.03\n",
        encoding="utf-8",
    )
    result = read_cobra(modern)
    assert result.normalized_s
    assert len(result.data) == 6
    assert np.isclose(result.summary()["maximum_growth_rate"], 0.05)
    assert np.isclose(result.summary()["unstable_fraction"], 2 / 6)
    assert result.summary()["failure_count"] == 0
    diagnosed, outputs = diagnose_cobra(modern, tmp_path / "cobra")
    assert diagnosed.summary()["field_line_count"] == 2
    assert outputs[0].stat().st_size > 1000

    legacy = tmp_path / "cobra_grate.legacy"
    legacy.write_text(
        "0.0 0.0 2\n"
        "2 -0.10\n"
        "3 0.02\n",
        encoding="utf-8",
    )
    old = read_cobra(legacy)
    assert not old.normalized_s
    assert list(old.data["s"]) == [2.0, 3.0]


def _write_executable(path: Path, source: str):
    path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_write_solver_inputs(tmp_path: Path):
    neo = write_neo_input(
        tmp_path / "neo_in.case",
        "boozmn_case.nc",
        "neo_out.case",
        [3, 7],
    )
    neo_lines = neo.read_text(encoding="utf-8").splitlines()
    assert neo_lines[3] == "boozmn_case.nc"
    assert neo_lines[4] == "neo_out.case"
    assert neo_lines[5:7] == ["2", "3 7"]

    cobra = write_cobra_input(
        tmp_path / "in_cobra.case",
        "case",
        [2, 5, 8],
        [0, 90],
        [0, 120, 240],
    )
    cobra_lines = cobra.read_text(encoding="utf-8").splitlines()
    assert cobra_lines == [
        "case",
        "10 1",
        "T F",
        "2",
        "0 90",
        "3",
        "0 120 240",
        "3",
        "2 5 8",
    ]
    with pytest.raises(ValueError, match="one-based"):
        write_cobra_input(
            tmp_path / "invalid_cobra.case",
            "case",
            [2],
            [0],
            [0],
            kth=0,
        )


def test_run_neo_solver_with_fake_executable(tmp_path: Path):
    wout = tmp_path / "wout_case.nc"
    xr.Dataset({"ns": xr.DataArray(9)}).to_netcdf(wout)
    boozmn = tmp_path / "boozmn_case.nc"
    xr.Dataset(
        {
            "jlist": ("radius", [2, 5, 8]),
            "s_b": ("radius", [0.125, 0.5, 0.875]),
            "bmnc_b": (("radius", "mn"), np.ones((3, 2))),
        }
    ).to_netcdf(boozmn)
    workdir = tmp_path / "neo_run/run"
    workdir.mkdir(parents=True)
    (workdir / "neo_param.case").write_text("stale extension control\n")
    (workdir / "neo_param.in").write_text("stale fallback control\n")
    executable = _write_executable(
        tmp_path / "xneo",
        "from pathlib import Path\n"
        "import sys\n"
        "ext = sys.argv[1]\n"
        "assert not Path(f'neo_param.{ext}').exists()\n"
        "assert not Path('neo_param.in').exists()\n"
        "control = Path(f'neo_in.{ext}').read_text().splitlines()\n"
        "assert control[5:7] == ['2', '1 2']\n"
        "Path(f'neo_out.{ext}').write_text("
        "'1 6e-5 0.4 0.74 2 5\\n2 8e-5 0.3 0.72 2 5\\n')\n",
    )
    result, run, outputs = run_neo_solver(
        wout,
        tmp_path / "neo_run",
        boozmn=boozmn,
        executable=executable,
        surface_indices=[8, 5],
    )
    assert run.returncode == 0
    assert result.summary()["surface_count"] == 2
    assert list(result.data["surface_label"]) == [8, 5]
    assert list(result.data["boozmn_surface_position"]) == [1, 2]
    with xr.open_dataset(tmp_path / "neo_run/run/boozmn_case.nc") as prepared:
        assert list(prepared["jlist"].values) == [1, 2]
        assert list(prepared["s_b"].values) == [0.875, 0.5]
    assert all(path.is_file() for path in outputs)


def test_stale_neo_output_cannot_mask_fortran_stop(tmp_path: Path):
    wout = tmp_path / "wout_case.nc"
    xr.Dataset({"ns": xr.DataArray(9)}).to_netcdf(wout)
    boozmn = tmp_path / "boozmn_case.nc"
    xr.Dataset(
        {
            "jlist": ("radius", [2]),
            "s_b": ("radius", [0.5]),
            "bmnc_b": (("radius", "mn"), np.ones((1, 2))),
        }
    ).to_netcdf(boozmn)
    stale_output = tmp_path / "neo_run/run/neo_out.case"
    stale_output.parent.mkdir(parents=True)
    stale_output.write_text("2 1e-4 0.2 0.7 2 5\n")
    executable = _write_executable(
        tmp_path / "xneo",
        "print('The requested surface is absent; intentional NEO stop')\n",
    )

    with pytest.raises(SolverExecutionError, match="intentional NEO stop"):
        run_neo_solver(
            wout,
            tmp_path / "neo_run",
            boozmn=boozmn,
            executable=executable,
        )
    assert not stale_output.exists()


def test_run_cobra_solver_with_fake_executable(tmp_path: Path):
    wout = tmp_path / "wout_case.nc"
    xr.Dataset({"ns": xr.DataArray(9)}).to_netcdf(wout)
    executable = _write_executable(
        tmp_path / "xcobravmec",
        "from pathlib import Path\n"
        "import sys\n"
        "input_path = Path(sys.argv[1])\n"
        "assert input_path.is_file() and sys.argv[2] == 'F'\n"
        "ext = input_path.name.split('.', 1)[1]\n"
        "Path(f'cobra_grate.{ext}').write_text("
        "'0 0 3\\n2 -0.1\\n5 -0.2\\n8 0.05\\n')\n",
    )
    result, run, outputs = run_cobra_solver(
        wout,
        tmp_path / "cobra_run",
        executable=executable,
        surface_indices=[2, 5, 8],
        ntheta=1,
        nzeta=1,
    )
    assert run.returncode == 0
    assert result.summary()["maximum_growth_rate"] == 0.05
    assert np.isclose(result.summary()["unstable_fraction"], 1 / 3)
    assert all(path.is_file() for path in outputs)


def test_cobra_failure_sentinel_is_not_physics(tmp_path: Path):
    source = tmp_path / "cobra_grate.failed"
    source.write_text(
        "0.0 0.0 2\n"
        "2 0.2 100.0\n"
        "3 0.4 -0.2\n",
        encoding="utf-8",
    )
    result = read_cobra(source)
    assert list(result.data["solver_failed"]) == [True, False]
    assert result.summary()["failure_count"] == 1
    assert result.summary()["valid_point_count"] == 1
    assert result.summary()["unstable_fraction"] == 0.0


def test_run_cobra_rejects_failure_sentinel(tmp_path: Path):
    wout = tmp_path / "wout_case.nc"
    xr.Dataset({"ns": xr.DataArray(9)}).to_netcdf(wout)
    executable = _write_executable(
        tmp_path / "xcobravmec",
        "from pathlib import Path\n"
        "import sys\n"
        "input_path = Path(sys.argv[1])\n"
        "ext = input_path.name.split('.', 1)[1]\n"
        "Path(f'cobra_grate.{ext}').write_text('0 0 1\\n2 0.2 100.0\\n')\n",
    )
    with pytest.raises(SolverExecutionError, match="failure sentinel"):
        run_cobra_solver(
            wout,
            tmp_path / "cobra_failed",
            executable=executable,
            surface_indices=[2],
            ntheta=1,
            nzeta=1,
        )


def test_missing_solver_executable_is_explicit(tmp_path: Path):
    wout = tmp_path / "wout_case.nc"
    xr.Dataset({"ns": xr.DataArray(9)}).to_netcdf(wout)
    try:
        run_cobra_solver(wout, tmp_path / "out", executable="definitely-not-xcobra")
    except SolverDependencyError as exc:
        assert "Source STELLOPT.sh" in str(exc)
    else:
        raise AssertionError("missing executable should raise SolverDependencyError")
