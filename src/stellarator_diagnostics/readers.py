"""Reader dispatch and file type detection."""

from __future__ import annotations

from pathlib import Path

from .desc_reader import DescAdapter
from .diagnostics import rational_surfaces
from .vmec import VmecAdapter


def detect_backend(path: str | Path) -> str:
    path = Path(path)
    name = path.name.lower()
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return "desc"
    if path.suffix.lower() == ".nc" and ("booz" in name or "boozmn" in name):
        return "boozer"
    if path.suffix.lower() == ".nc":
        return "vmec"
    raise ValueError(f"Cannot infer equilibrium type from {path}")


def load_equilibrium(path: str | Path, backend="auto", family_index=-1):
    backend = detect_backend(path) if backend == "auto" else backend.lower()
    if backend == "vmec":
        eq = VmecAdapter(path).to_data()
    elif backend == "desc":
        eq = DescAdapter(path, family_index=family_index).to_data()
    else:
        raise ValueError("Boozer files are field-map supplements, not complete equilibria")
    if "iota" in eq.profiles:
        eq.metadata["low_order_rational_surfaces"] = rational_surfaces(*eq.profiles["iota"])
    return eq
