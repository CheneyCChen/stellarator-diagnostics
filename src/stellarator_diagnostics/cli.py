"""Command-line interface."""

from __future__ import annotations

import argparse
import glob
import json

from .api import analyze, compare, scan
from .boozer import run_booz_xform
from .external import diagnose_cobra, diagnose_dkes, diagnose_neo
from .plots import plot_boozer_surface_files
from .readers import load_equilibrium


def _parser():
    parser = argparse.ArgumentParser(
        prog="stell-diag",
        description="Unified VMEC/DESC equilibrium diagnostics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Generate a complete report for one equilibrium")
    p.add_argument("file")
    p.add_argument("-o", "--outdir")
    p.add_argument("--backend", choices=["auto", "vmec", "desc"], default="auto")
    p.add_argument("--family-index", type=int, default=-1)
    p.add_argument("--surface", type=float, default=1.0)
    p.add_argument("--boozmn", help="Optional BOOZ_XFORM boozmn NetCDF")
    p.add_argument("--neo-out", help="Optional STELLOPT NEO neo_out file")
    p.add_argument("--dkes-results", help="Optional STELLOPT DKES results file")
    p.add_argument("--cobra-grate", help="Optional COBRAVMEC cobra_grate file")

    p = sub.add_parser("compare", help="Compare two or more equilibria")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--outdir", default="comparison")
    p.add_argument("--backend", choices=["auto", "vmec", "desc"], default="auto")

    p = sub.add_parser("scan", help="Create a scalar table from many files")
    p.add_argument("patterns", nargs="+", help="File paths or glob patterns")
    p.add_argument("-o", "--output", default="scan_summary.csv")
    p.add_argument("--backend", choices=["auto", "vmec", "desc"], default="auto")

    p = sub.add_parser("boozer", help="Plot unfilled |B| contours from a boozmn NetCDF file")
    p.add_argument("file")
    p.add_argument("-o", "--output", default="boozer_surfaces")
    p.add_argument(
        "--surfaces",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0],
    )

    p = sub.add_parser(
        "xboozer",
        help="Run maintained booz_xform on a wout and write boozmn NetCDF",
    )
    p.add_argument("wout")
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--surfaces",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0],
    )
    p.add_argument("--mboz", type=int)
    p.add_argument("--nboz", type=int)

    p = sub.add_parser("neo", help="Diagnose a STELLOPT NEO neo_out file")
    p.add_argument("file")
    p.add_argument("-o", "--outdir", default="neo_diagnostics")

    p = sub.add_parser("dkes", help="Diagnose a STELLOPT DKES results file")
    p.add_argument("file")
    p.add_argument("-o", "--outdir", default="dkes_diagnostics")

    p = sub.add_parser("cobra", help="Diagnose a COBRAVMEC cobra_grate file")
    p.add_argument("file")
    p.add_argument("-o", "--outdir", default="cobra_diagnostics")

    p = sub.add_parser("summary", help="Print scalar diagnostics as JSON")
    p.add_argument("file")
    p.add_argument("--backend", choices=["auto", "vmec", "desc"], default="auto")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "analyze":
        eq, report = analyze(
            args.file,
            args.outdir,
            backend=args.backend,
            family_index=args.family_index,
            surface_s=args.surface,
            boozmn=args.boozmn,
            neo_out=args.neo_out,
            dkes_results=args.dkes_results,
            cobra_grate=args.cobra_grate,
        )
        print(report)
        for warning in eq.warnings:
            print(f"WARNING: {warning}")
    elif args.command == "compare":
        _, table = compare(args.files, args.outdir, backend=args.backend)
        print(table.to_string(index=False))
    elif args.command == "scan":
        files = []
        for pattern in args.patterns:
            matches = glob.glob(pattern)
            files.extend(matches or [pattern])
        table, errors = scan(files, args.output, backend=args.backend)
        print(table.to_string(index=False))
        if errors:
            print(f"{len(errors)} file(s) failed; see scan_errors.csv")
    elif args.command == "boozer":
        outputs = plot_boozer_surface_files(args.file, args.output, surfaces=args.surfaces)
        print("\n".join(str(path) for path in outputs))
    elif args.command == "xboozer":
        print(
            run_booz_xform(
                args.wout,
                args.output,
                surfaces=args.surfaces,
                mboz=args.mboz,
                nboz=args.nboz,
            )
        )
    elif args.command in {"neo", "dkes", "cobra"}:
        diagnostic = {
            "neo": diagnose_neo,
            "dkes": diagnose_dkes,
            "cobra": diagnose_cobra,
        }[args.command]
        result, outputs = diagnostic(args.file, args.outdir)
        print(json.dumps(result.summary(), indent=2, default=str))
        print("\n".join(str(path) for path in outputs))
    elif args.command == "summary":
        eq = load_equilibrium(args.file, backend=args.backend)
        print(json.dumps(eq.scalar_row(), indent=2, default=str))


if __name__ == "__main__":
    main()
