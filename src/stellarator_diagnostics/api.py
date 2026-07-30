"""High-level public API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .boozer import infer_boozer_resolution, run_booz_xform
from .plots import plot_comparison
from .readers import load_equilibrium
from .report import write_report


def analyze(
    path,
    outdir=None,
    backend="auto",
    family_index=-1,
    surface_s=1.0,
    boozmn=None,
    neo_out=None,
    dkes_results=None,
    cobra_grate=None,
    mboz=None,
    nboz=None,
):
    eq = load_equilibrium(path, backend=backend, family_index=family_index)
    outdir = Path(outdir or f"diagnostics_{eq.label}")
    outdir.mkdir(parents=True, exist_ok=True)
    if boozmn is None and eq.backend == "VMEC":
        generated = outdir / f"boozmn_{eq.label}.nc"
        try:
            inferred_mboz, inferred_nboz = infer_boozer_resolution(path)
            actual_mboz = inferred_mboz if mboz is None else int(mboz)
            actual_nboz = inferred_nboz if nboz is None else int(nboz)
            boozmn = run_booz_xform(
                path,
                generated,
                mboz=actual_mboz,
                nboz=actual_nboz,
            )
            eq.metadata["boozer_mboz"] = actual_mboz
            eq.metadata["boozer_nboz"] = actual_nboz
            eq.metadata["boozmn"] = str(boozmn)
        except (ImportError, KeyError, RuntimeError, OSError, ValueError) as exc:
            eq.warnings.append(
                "automatic Boozer transform skipped: "
                f"{type(exc).__name__}: {exc}"
            )
    report = write_report(
        eq,
        outdir,
        surface_s=surface_s,
        boozmn=boozmn,
        neo_out=neo_out,
        dkes_results=dkes_results,
        cobra_grate=cobra_grate,
    )
    return eq, report


def compare(paths, outdir="comparison", backend="auto"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    equilibria = [load_equilibrium(path, backend=backend) for path in paths]
    table = pd.DataFrame([eq.scalar_row() for eq in equilibria])
    table.to_csv(outdir / "comparison.csv", index=False)
    plot_comparison(equilibria, outdir / "profiles_comparison.png")
    return equilibria, table


def scan(paths, output="scan_summary.csv", backend="auto"):
    rows = []
    errors = []
    for path in paths:
        eq = None
        try:
            eq = load_equilibrium(path, backend=backend)
            rows.append(eq.scalar_row())
        except Exception as exc:
            errors.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if eq is not None:
                eq.close()
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(Path(output).with_name("scan_errors.csv"), index=False)
    return table, errors
