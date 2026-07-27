"""Example: use the Python API and add a project-specific acceptance test."""

from pathlib import Path

from stellarator_diagnostics import analyze

eq, report = analyze(
    "wout_pwO_nfp4_final.nc",
    outdir=Path("results") / "pwO_nfp4",
    surface_s=0.5,
)

checks = {
    "aspect near 8": abs(eq.scalars["aspect"] - 8.0) < 0.2,
    "iota axis": 0.67 < eq.scalars["iota_axis"] < 0.70,
    "iota edge": 0.77 < eq.scalars["iota_edge"] < 0.81,
    "Mercier stable outside axis": eq.scalars.get("D_Mercier_min_s>=0.05", float("-inf")) >= 0,
}
print(report)
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
