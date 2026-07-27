"""Safe subprocess runners for external STELLOPT diagnostics."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from .boozer import infer_boozer_resolution, run_booz_xform
from .external import CobraResult, NeoResult, diagnose_cobra, diagnose_neo


class SolverDependencyError(RuntimeError):
    """Raised when a requested STELLOPT executable cannot be found."""


class SolverExecutionError(RuntimeError):
    """Raised when an external solver fails or produces invalid output."""


@dataclass(slots=True)
class SolverRun:
    """Machine-readable record of one external solver invocation."""

    solver: str
    executable: str
    command: list[str]
    workdir: str
    input_file: str
    output_file: str
    returncode: int
    elapsed_seconds: float
    stdout_log: str
    stderr_log: str


def _resolve_executable(requested: str | Path, solver: str) -> Path:
    candidate = Path(requested).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise SolverDependencyError(f"{solver} executable does not exist: {resolved}")
    else:
        found = shutil.which(str(requested))
        if found is None:
            raise SolverDependencyError(
                f"Cannot find `{requested}` for {solver}. Source STELLOPT.sh or pass "
                f"`--executable /absolute/path/to/{candidate.name}`."
            )
        resolved = Path(found).resolve()
    if not resolved.stat().st_mode & 0o111:
        raise SolverDependencyError(f"{solver} executable is not executable: {resolved}")
    return resolved


def _case_extension(wout: str | Path) -> str:
    name = Path(wout).name
    if name.startswith("wout_"):
        name = name[5:]
    if name.endswith(".nc"):
        name = name[:-3]
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in name)
    if not safe:
        raise ValueError(f"Cannot derive a solver extension from {wout}")
    return safe


def _copy_as(source: str | Path, destination: Path) -> Path:
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _run_command(
    solver: str,
    executable: Path,
    arguments: Sequence[str],
    workdir: Path,
    input_file: Path,
    output_file: Path,
    timeout: float,
) -> SolverRun:
    command = [str(executable), *map(str, arguments)]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_log = workdir / f"{solver.lower()}_stdout.log"
        stderr_log = workdir / f"{solver.lower()}_stderr.log"
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        stdout_log.write_text(stdout or "", encoding="utf-8")
        stderr_log.write_text(stderr or "", encoding="utf-8")
        raise SolverExecutionError(
            f"{solver} exceeded timeout={timeout:g} s; partial logs are retained in {workdir}"
        ) from exc
    elapsed = time.monotonic() - started
    stdout_log = workdir / f"{solver.lower()}_stdout.log"
    stderr_log = workdir / f"{solver.lower()}_stderr.log"
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise SolverExecutionError(
            f"{solver} failed with return code {completed.returncode}. "
            f"See {stdout_log} and {stderr_log}."
        )
    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise SolverExecutionError(
            f"{solver} returned success but did not create a non-empty {output_file.name}"
        )
    record = SolverRun(
        solver=solver,
        executable=str(executable),
        command=command,
        workdir=str(workdir),
        input_file=str(input_file),
        output_file=str(output_file),
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )
    (workdir / f"{solver.lower()}_run.json").write_text(
        json.dumps(asdict(record), indent=2),
        encoding="utf-8",
    )
    return record


def _boozmn_surface_indices(boozmn: str | Path) -> list[int]:
    with xr.open_dataset(boozmn, decode_cf=False, mask_and_scale=False) as ds:
        if "jlist" in ds:
            values = np.ravel(np.asarray(ds["jlist"].values, dtype=int))
        elif "s_b" in ds:
            values = np.arange(1, np.asarray(ds["s_b"]).size + 1, dtype=int)
        elif "bmnc_b" in ds:
            shape = np.asarray(ds["bmnc_b"]).shape
            values = np.arange(1, min(shape) + 1, dtype=int)
        else:
            raise ValueError("Cannot determine NEO surfaces from boozmn")
    return sorted({int(value) for value in values if int(value) > 0})


def write_neo_input(
    path: str | Path,
    boozmn_name: str,
    output_name: str,
    surface_indices: Sequence[int],
    theta_n: int = 200,
    phi_n: int = 200,
    npart: int = 75,
    multra: int = 1,
    accuracy: float = 0.01,
    nstep_min: int = 500,
    nstep_max: int = 5000,
) -> Path:
    """Write the official standalone NEO text-control format."""
    surfaces = [int(value) for value in surface_indices]
    if not surfaces:
        raise ValueError("NEO requires at least one surface index")
    if theta_n <= 100 or phi_n <= 100:
        raise ValueError("NEO theta_n and phi_n must both be greater than 100")
    lines = [
        "# stellarator-diagnostics generated NEO input",
        "# standalone xneo text format",
        "# values follow STELLOPT NEO documentation",
        boozmn_name,
        output_name,
        str(len(surfaces)),
        " ".join(map(str, surfaces)),
        str(theta_n),
        str(phi_n),
        "0",  # max_m_mode: use all available modes
        "0",  # max_n_mode: use all available modes
        str(npart),
        str(multra),
        f"{accuracy:.12g}",
        "100",  # no_bins
        "75",  # nstep_per
        str(nstep_min),
        str(nstep_max),
        "0",  # calc_nstep_max
        "1",  # eout_swi: standard six-column neo_out
        "0",  # lab_swi
        "0",  # inp_swi: BOOZ_XFORM input
        "2",  # ref_swi: maximum B on each surface
        "0",  # write_progress
        "0",  # write_output_files
        "0",  # spline_test
        "0",  # write_integrate
        "0",  # write_diagnostic
        "# current calculation disabled",
        "#",
        "#",
        "0",  # calc_cur
        "neo_cur",
        "0",  # npart_cur
        "0",  # alpha_cur
        "0",  # write_cur_inte
    ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_neo_solver(
    wout: str | Path,
    outdir: str | Path,
    boozmn: str | Path | None = None,
    executable: str | Path = "xneo",
    surface_indices: Sequence[int] | None = None,
    mboz: int | None = None,
    nboz: int | None = None,
    theta_n: int = 200,
    phi_n: int = 200,
    npart: int = 75,
    multra: int = 1,
    accuracy: float = 0.01,
    nstep_min: int = 500,
    nstep_max: int = 5000,
    timeout: float = 3600,
) -> tuple[NeoResult, SolverRun, list[Path]]:
    """Run standalone STELLOPT NEO and diagnose the verified output."""
    executable_path = _resolve_executable(executable, "NEO")
    outdir = Path(outdir).resolve()
    workdir = outdir / "run"
    workdir.mkdir(parents=True, exist_ok=True)
    extension = _case_extension(wout)
    _copy_as(wout, workdir / f"wout_{extension}.nc")
    local_booz = workdir / f"boozmn_{extension}.nc"
    if boozmn is None:
        actual_mboz, actual_nboz = infer_boozer_resolution(wout)
        run_booz_xform(
            wout,
            local_booz,
            mboz=actual_mboz if mboz is None else mboz,
            nboz=actual_nboz if nboz is None else nboz,
        )
    else:
        _copy_as(boozmn, local_booz)
    available = _boozmn_surface_indices(local_booz)
    surfaces = available if surface_indices is None else [int(value) for value in surface_indices]
    missing = sorted(set(surfaces) - set(available))
    if missing:
        raise ValueError(f"NEO surfaces are absent from boozmn jlist: {missing}")
    input_file = write_neo_input(
        workdir / f"neo_in.{extension}",
        local_booz.name,
        f"neo_out.{extension}",
        surfaces,
        theta_n=theta_n,
        phi_n=phi_n,
        npart=npart,
        multra=multra,
        accuracy=accuracy,
        nstep_min=nstep_min,
        nstep_max=nstep_max,
    )
    output_file = workdir / f"neo_out.{extension}"
    record = _run_command(
        "NEO",
        executable_path,
        [extension],
        workdir,
        input_file,
        output_file,
        timeout,
    )
    result, figures = diagnose_neo(output_file, outdir / "diagnostics")
    return result, record, figures


def _vmec_ns(wout: str | Path) -> int:
    with xr.open_dataset(wout, decode_cf=False, mask_and_scale=False) as ds:
        if "ns" not in ds:
            raise KeyError("VMEC wout is missing ns")
        return int(np.asarray(ds["ns"]).item())


def _even_surface_indices(ns: int, count: int) -> list[int]:
    if ns < 3:
        raise ValueError("COBRAVMEC requires a VMEC equilibrium with ns >= 3")
    count = max(1, min(int(count), ns - 2))
    return sorted({int(value) for value in np.rint(np.linspace(2, ns - 1, count))})


def write_cobra_input(
    path: str | Path,
    extension: str,
    surface_indices: Sequence[int],
    zeta_degrees: Sequence[float],
    theta_degrees: Sequence[float],
    k_w: int = 10,
    kth: int = 0,
) -> Path:
    """Write the COBRAVMEC v4.1 input format from official documentation."""
    surfaces = [int(value) for value in surface_indices]
    zeta = [float(value) for value in zeta_degrees]
    theta = [float(value) for value in theta_degrees]
    if not surfaces or not zeta or not theta:
        raise ValueError("COBRAVMEC surfaces and starting-angle arrays cannot be empty")
    lines = [
        extension,
        f"{int(k_w)} {int(kth)}",
        "T F",  # use VMEC geometry and theta/zeta starting angles
        str(len(zeta)),
        " ".join(f"{value:.12g}" for value in zeta),
        str(len(theta)),
        " ".join(f"{value:.12g}" for value in theta),
        str(len(surfaces)),
        " ".join(map(str, surfaces)),
    ]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_cobra_solver(
    wout: str | Path,
    outdir: str | Path,
    executable: str | Path = "xcobravmec",
    surface_indices: Sequence[int] | None = None,
    nsurfaces: int = 16,
    ntheta: int = 5,
    nzeta: int = 5,
    k_w: int = 10,
    kth: int = 0,
    timeout: float = 7200,
) -> tuple[CobraResult, SolverRun, list[Path]]:
    """Run STELLOPT COBRAVMEC v4.1 and diagnose the verified output."""
    executable_path = _resolve_executable(executable, "COBRAVMEC")
    outdir = Path(outdir).resolve()
    workdir = outdir / "run"
    workdir.mkdir(parents=True, exist_ok=True)
    extension = _case_extension(wout)
    _copy_as(wout, workdir / f"wout_{extension}.nc")
    ns = _vmec_ns(wout)
    surfaces = (
        _even_surface_indices(ns, nsurfaces)
        if surface_indices is None
        else sorted({int(value) for value in surface_indices})
    )
    if any(value < 2 or value >= ns for value in surfaces):
        raise ValueError(f"COBRAVMEC surface indices must satisfy 2 <= index < ns={ns}")
    if ntheta < 1 or nzeta < 1:
        raise ValueError("COBRAVMEC ntheta and nzeta must be positive")
    theta = np.linspace(0, 360, ntheta, endpoint=False)
    zeta = np.linspace(0, 360, nzeta, endpoint=False)
    input_file = write_cobra_input(
        workdir / f"in_cobra.{extension}",
        extension,
        surfaces,
        zeta,
        theta,
        k_w=k_w,
        kth=kth,
    )
    output_file = workdir / f"cobra_grate.{extension}"
    record = _run_command(
        "COBRAVMEC",
        executable_path,
        [input_file.name, "F"],
        workdir,
        input_file,
        output_file,
        timeout,
    )
    result, figures = diagnose_cobra(output_file, outdir / "diagnostics")
    return result, record, figures
