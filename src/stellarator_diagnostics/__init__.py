"""Unified diagnostics for VMEC and DESC equilibria."""

from .api import analyze, compare, scan
from .external import read_cobra, read_dkes, read_neo
from .model import EquilibriumData, FieldMap, Surface
from .runners import run_cobra_solver, run_neo_solver

__all__ = [
    "EquilibriumData",
    "FieldMap",
    "Surface",
    "analyze",
    "compare",
    "read_cobra",
    "read_dkes",
    "read_neo",
    "run_cobra_solver",
    "run_neo_solver",
    "scan",
]
__version__ = "0.4.1"
