"""Command-line interface."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict

from .api import analyze, compare, scan
from .boozer import run_booz_xform
from .external import diagnose_cobra, diagnose_dkes, diagnose_neo
from .plots import plot_boozer_surface_files
from .readers import load_equilibrium
from .runners import DEFAULT_DKES_CMUL, run_cobra_solver, run_dkes_solver, run_neo_solver


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
    p.add_argument("--mboz", type=int, help="Override automatically inferred BOOZ_XFORM m")
    p.add_argument("--nboz", type=int, help="Override automatically inferred BOOZ_XFORM n")
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

    p = sub.add_parser("run-neo", help="Run STELLOPT xneo and diagnose its output")
    p.add_argument("wout")
    p.add_argument("-o", "--outdir", default="neo_run")
    p.add_argument("--boozmn", help="Existing boozmn; otherwise run BOOZ_XFORM")
    p.add_argument("--executable", default="xneo")
    p.add_argument(
        "--surface-indices",
        type=int,
        nargs="+",
        help="VMEC/BOOZ_XFORM jlist labels; the boozmn is staged unchanged",
    )
    p.add_argument("--mboz", type=int)
    p.add_argument("--nboz", type=int)
    p.add_argument("--theta-n", type=int, default=200)
    p.add_argument("--phi-n", type=int, default=200)
    p.add_argument("--npart", type=int, default=75)
    p.add_argument("--multra", type=int, default=1)
    p.add_argument("--accuracy", type=float, default=0.01)
    p.add_argument("--nstep-min", type=int, default=500)
    p.add_argument("--nstep-max", type=int, default=5000)
    p.add_argument("--timeout", type=float, default=3600)

    p = sub.add_parser(
        "run-dkes",
        help="Run STELLOPT xdkes and scan the radial monoenergetic coefficient D11*",
    )
    p.add_argument("wout")
    p.add_argument("-o", "--outdir", default="dkes_run")
    p.add_argument("--boozmn", help="Existing boozmn; otherwise run BOOZ_XFORM")
    p.add_argument("--executable", default="xdkes")
    p.add_argument(
        "--surface-indices",
        type=int,
        nargs="+",
        help="VMEC/BOOZ_XFORM jlist labels; defaults to every available Boozer surface",
    )
    p.add_argument(
        "--cmul",
        type=float,
        nargs="+",
        default=list(DEFAULT_DKES_CMUL),
        help="Collisionality scan values nu/v [m^-1]",
    )
    p.add_argument(
        "--efield",
        type=float,
        nargs="+",
        default=[0.0],
        help="Normalized radial electric-field values E_s/v",
    )
    p.add_argument("--mboz", type=int)
    p.add_argument("--nboz", type=int)
    p.add_argument("--coupling-order", type=int, default=4)
    p.add_argument("--lalpha", type=int, default=100)
    p.add_argument("--timeout", type=float, default=7200, help="Timeout per surface [s]")

    p = sub.add_parser(
        "run-cobra",
        help="Run STELLOPT COBRAVMEC v4.1 and diagnose its output",
    )
    p.add_argument("wout")
    p.add_argument("-o", "--outdir", default="cobra_run")
    p.add_argument("--executable", default="xcobravmec")
    p.add_argument("--surface-indices", type=int, nargs="+")
    p.add_argument("--nsurfaces", type=int, default=16)
    p.add_argument("--ntheta", type=int, default=5)
    p.add_argument("--nzeta", type=int, default=5)
    p.add_argument("--k-w", type=int, default=10)
    p.add_argument(
        "--kth",
        type=int,
        default=1,
        help="One-based ballooning mode label (1 = most unstable)",
    )
    p.add_argument("--timeout", type=float, default=7200)

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
            mboz=args.mboz,
            nboz=args.nboz,
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
    elif args.command == "run-neo":
        result, run, outputs = run_neo_solver(
            args.wout,
            args.outdir,
            boozmn=args.boozmn,
            executable=args.executable,
            surface_indices=args.surface_indices,
            mboz=args.mboz,
            nboz=args.nboz,
            theta_n=args.theta_n,
            phi_n=args.phi_n,
            npart=args.npart,
            multra=args.multra,
            accuracy=args.accuracy,
            nstep_min=args.nstep_min,
            nstep_max=args.nstep_max,
            timeout=args.timeout,
        )
        print(json.dumps({"run": asdict(run), "summary": result.summary()}, indent=2, default=str))
        print("\n".join(str(path) for path in outputs))
    elif args.command == "run-dkes":
        result, runs, outputs = run_dkes_solver(
            args.wout,
            args.outdir,
            boozmn=args.boozmn,
            executable=args.executable,
            surface_indices=args.surface_indices,
            cmul=args.cmul,
            efield=args.efield,
            mboz=args.mboz,
            nboz=args.nboz,
            coupling_order=args.coupling_order,
            lalpha=args.lalpha,
            timeout=args.timeout,
        )
        print(
            json.dumps(
                {"runs": [asdict(run) for run in runs], "summary": result.summary()},
                indent=2,
                default=str,
            )
        )
        print("\n".join(str(path) for path in outputs))
    elif args.command == "run-cobra":
        result, run, outputs = run_cobra_solver(
            args.wout,
            args.outdir,
            executable=args.executable,
            surface_indices=args.surface_indices,
            nsurfaces=args.nsurfaces,
            ntheta=args.ntheta,
            nzeta=args.nzeta,
            k_w=args.k_w,
            kth=args.kth,
            timeout=args.timeout,
        )
        print(json.dumps({"run": asdict(run), "summary": result.summary()}, indent=2, default=str))
        print("\n".join(str(path) for path in outputs))
    elif args.command == "summary":
        eq = load_equilibrium(args.file, backend=args.backend)
        print(json.dumps(eq.scalar_row(), indent=2, default=str))


if __name__ == "__main__":
    main()
