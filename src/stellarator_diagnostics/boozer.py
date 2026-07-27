"""Integration with the mature hiddenSymmetries BOOZ_XFORM implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_booz_xform(
    wout: str | Path,
    output: str | Path,
    surfaces=(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0),
    mboz: int | None = None,
    nboz: int | None = None,
):
    """Transform a VMEC wout using the maintained ``booz_xform`` Python API."""
    try:
        import booz_xform as bx
    except ImportError as exc:
        raise ImportError(
            "Running the Boozer transform requires the maintained `booz_xform` "
            "package (or run STELLOPT `xbooz_xform` externally and pass its "
            "boozmn_*.nc output to `stell-diag boozer`)."
        ) from exc

    transform = bx.Booz_xform()
    transform.read_wout(str(wout))
    transform.verbose = False
    if mboz is not None:
        transform.mboz = int(mboz)
    if nboz is not None:
        transform.nboz = int(nboz)
    s_in = np.asarray(transform.s_in, dtype=float)
    indices = sorted(
        {int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces if 0 <= float(s) <= 1}
    )
    transform.compute_surfs = indices
    transform.run()
    transform.write_boozmn(str(output))
    return Path(output)
