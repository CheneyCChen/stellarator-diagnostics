"""Integration with the mature hiddenSymmetries BOOZ_XFORM implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def infer_boozer_resolution(wout: str | Path) -> tuple[int, int]:
    """Infer BOOZ_XFORM truncation from the Fourier modes stored by VMEC.

    VMEC stores ``xn`` including the field-period factor, so the physical
    toroidal mode index is ``abs(xn) / nfp``.  Prefer the Nyquist arrays
    because ``|B|`` is represented on that grid.  This avoids fixed,
    case-independent BOOZ_XFORM truncations.
    """
    with xr.open_dataset(wout, decode_cf=False, mask_and_scale=False) as ds:
        nfp = int(np.asarray(ds["nfp"]).item())
        xm_name = "xm_nyq" if "xm_nyq" in ds else "xm"
        xn_name = "xn_nyq" if "xn_nyq" in ds else "xn"
        if xm_name not in ds or xn_name not in ds:
            raise KeyError("VMEC wout is missing xm/xn Fourier mode arrays")
        xm_max = float(np.nanmax(np.abs(np.asarray(ds[xm_name].values, dtype=float))))
        xn_max = float(np.nanmax(np.abs(np.asarray(ds[xn_name].values, dtype=float))))
        mpol = int(np.asarray(ds["mpol"]).item()) if "mpol" in ds else 1
        ntor = int(np.asarray(ds["ntor"]).item()) if "ntor" in ds else 0
    mboz = max(mpol, int(np.ceil(xm_max)))
    nboz = max(ntor, int(np.ceil(xn_max / max(nfp, 1))))
    return mboz, nboz


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
    inferred_mboz, inferred_nboz = infer_boozer_resolution(wout)
    transform.mboz = int(inferred_mboz if mboz is None else mboz)
    transform.nboz = int(inferred_nboz if nboz is None else nboz)
    s_in = np.asarray(transform.s_in, dtype=float)
    indices = sorted(
        {int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces if 0 <= float(s) <= 1}
    )
    transform.compute_surfs = indices
    transform.run()
    transform.write_boozmn(str(output))
    return Path(output)
