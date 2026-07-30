"""Unified diagnostics for VMEC and DESC equilibria."""

from .api import analyze, compare, scan
from .external import read_cobra, read_dkes, read_neo
from .model import EquilibriumData, FieldMap, Surface
from .qi import compute_goodman_qi, diagnose_goodman_qi
from .runners import run_cobra_solver, run_dkes_solver, run_neo_solver

__all__ = [
    "EquilibriumData",
    "FieldMap",
    "Surface",
    "analyze",
    "compare",
    "compute_goodman_qi",
    "diagnose_goodman_qi",
    "read_cobra",
    "read_dkes",
    "read_neo",
    "run_cobra_solver",
    "run_dkes_solver",
    "run_neo_solver",
    "scan",
]
__version__ = "0.6.1"
