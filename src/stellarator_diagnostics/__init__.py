"""Unified diagnostics for VMEC and DESC equilibria."""

from .api import analyze, compare, scan
from .external import read_cobra, read_dkes, read_neo
from .model import EquilibriumData, FieldMap, Surface

__all__ = [
    "EquilibriumData",
    "FieldMap",
    "Surface",
    "analyze",
    "compare",
    "read_cobra",
    "read_dkes",
    "read_neo",
    "scan",
]
__version__ = "0.3.0"
