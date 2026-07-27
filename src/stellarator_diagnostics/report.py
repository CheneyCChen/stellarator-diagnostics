"""Report generation and machine-readable exports."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external import diagnose_cobra, diagnose_dkes, diagnose_neo
from .model import EquilibriumData
from .plots import (
    plot_boundary_angles,
    plot_boozer_surface_files,
    plot_cross_sections,
    plot_fieldline_traces,
    plot_iota,
    plot_long_fieldline_trace,
    plot_profiles,
    plot_mercier_terms,
    plot_mercier_total,
    plot_surface_3d,
)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def write_report(
    eq: EquilibriumData,
    outdir: str | Path,
    surface_s=1.0,
    boozmn: str | Path | None = None,
    neo_out: str | Path | None = None,
    dkes_results: str | Path | None = None,
    cobra_grate: str | Path | None = None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = {}

    def surface_3d_job(view, filename):
        surface = eq.surface(s=surface_s, ntheta=160, nphi=160)
        field = eq.field_map(s=surface_s, ntheta=160, nzeta=160)
        return plot_surface_3d(
            surface,
            outdir / filename,
            field_values=field.values,
            nfp=eq.nfp,
            view=view,
        )

    jobs = [
        ("iota", lambda: plot_iota(eq, outdir / "iota.png")),
        ("profiles", lambda: plot_profiles(eq, outdir / "profiles.png")),
        (
            "mercier_total",
            lambda: plot_mercier_total(eq, outdir / "mercier_total.png"),
        ),
        (
            "mercier_terms",
            lambda: plot_mercier_terms(eq, outdir / "mercier_terms.png"),
        ),
        ("cross_sections", lambda: plot_cross_sections(eq, outdir / "cross_sections.png")),
        (
            "boundary_angles",
            lambda: plot_boundary_angles(eq, outdir / "boundary_angles.png"),
        ),
        (
            "fieldline_traces",
            lambda: plot_fieldline_traces(eq, outdir / "fieldline_traces.png", s=surface_s),
        ),
        (
            "fieldline_long",
            lambda: plot_long_fieldline_trace(
                eq,
                outdir / "fieldline_long.png",
                s=surface_s,
                alpha_pi=0,
                periods=200,
            ),
        ),
        (
            "surface_3d",
            lambda: surface_3d_job("perspective", "surface_3d.png"),
        ),
        (
            "surface_top",
            lambda: surface_3d_job("top", "surface_top.png"),
        ),
    ]
    for name, job in jobs:
        try:
            result = job()
            if result is not None:
                figures[name] = result.name
        except Exception as exc:
            eq.warnings.append(f"{name} skipped: {type(exc).__name__}: {exc}")
    if boozmn is not None:
        try:
            outputs = plot_boozer_surface_files(boozmn, outdir / "boozer")
            for output in outputs:
                figures[output.stem] = str(output.relative_to(outdir))
        except Exception as exc:
            eq.warnings.append(f"boozer surfaces skipped: {type(exc).__name__}: {exc}")

    external_diagnostics = {}
    external_jobs = [
        ("neo", neo_out, diagnose_neo),
        ("dkes", dkes_results, diagnose_dkes),
        ("cobra", cobra_grate, diagnose_cobra),
    ]
    for name, source, diagnostic in external_jobs:
        if source is None:
            continue
        try:
            result, outputs = diagnostic(source, outdir / name)
            external_diagnostics[name] = result.summary()
            for output in outputs:
                figures[output.stem] = str(output.relative_to(outdir))
        except Exception as exc:
            eq.warnings.append(f"{name} skipped: {type(exc).__name__}: {exc}")

    payload = {
        "label": eq.label,
        "backend": eq.backend,
        "source": str(eq.source),
        "nfp": eq.nfp,
        "scalars": eq.scalars,
        "metadata": eq.metadata,
        "warnings": eq.warnings,
        "profiles": {
            name: {"s": s, "value": value, "units": eq.profile_units.get(name, "")}
            for name, (s, value) in eq.profiles.items()
        },
        "stability": {name: {"s": s, "value": value} for name, (s, value) in eq.stability.items()},
        "external_diagnostics": external_diagnostics,
    }
    (outdir / "diagnostics.json").write_text(
        json.dumps(payload, cls=JsonEncoder, indent=2), encoding="utf-8"
    )
    pd.DataFrame([eq.scalar_row()]).to_csv(outdir / "summary.csv", index=False)
    for name, (s, value) in {**eq.profiles, **eq.stability}.items():
        pd.DataFrame({"s": s, name: value}).to_csv(outdir / f"{name}.csv", index=False)

    report_values = eq.scalar_row()
    for diagnostic, summary in external_diagnostics.items():
        for key, value in summary.items():
            if key != "source":
                report_values[f"{diagnostic}.{key}"] = value
    rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(_format_value(v))}</td></tr>"
        for k, v in report_values.items()
    )
    images = "\n".join(
        f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2>"
        f"<img src='{html.escape(filename)}' alt='{html.escape(name)}'></section>"
        for name, filename in figures.items()
    )
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in eq.warnings)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(eq.label)} diagnostics</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}
img{{max-width:100%;height:auto}}section{{margin:2rem 0}}.warning{{color:#9a5b00}}
</style></head><body>
<h1>{html.escape(eq.label)} — {eq.backend} diagnostics</h1>
<table>{rows}</table>
<ul class="warning">{warnings}</ul>
{images}
</body></html>"""
    (outdir / "report.html").write_text(page, encoding="utf-8")
    return outdir / "report.html"


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
