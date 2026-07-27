"""High-level public API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
):
    eq = load_equilibrium(path, backend=backend, family_index=family_index)
    outdir = Path(outdir or f"diagnostics_{eq.label}")
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
        try:
            rows.append(load_equilibrium(path, backend=backend).scalar_row())
        except Exception as exc:
            errors.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(Path(output).with_name("scan_errors.csv"), index=False)
    return table, errors
