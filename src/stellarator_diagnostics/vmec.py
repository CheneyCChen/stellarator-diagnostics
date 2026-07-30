"""VMEC and BOOZ_XFORM readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .mathutils import (
    as_scalar,
    edge_extrapolate,
    fourier_cos,
    fourier_sin,
    interp_radial,
    radial_grid,
)
from .model import EquilibriumData, FieldMap, Surface

PROFILE_ALIASES = {
    "iota": ("iotaf", "iotas"),
    "pressure": ("presf", "pres"),
    "mass": ("mass",),
    "toroidal_current": ("jcurv", "jcuru"),
    "dV_ds": ("vp",),
    "beta": ("betaxis",),
}

STABILITY_NAMES = {
    "D_Mercier": ("DMerc",),
    "D_shear": ("DShear", "Dshear"),
    "D_well": ("DWell", "Dwell"),
    "D_current": ("DCurr", "Dcurr"),
    "D_geodesic": ("DGeod", "Dgeod"),
}


class VmecAdapter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds = xr.open_dataset(self.path, decode_cf=False, mask_and_scale=False)
        self.ns = int(as_scalar(self._get("ns"), self.ds.sizes.get("radius", 1)))
        self.nfp = int(as_scalar(self._get("nfp"), 1))
        self.lasym = bool(as_scalar(self._get("lasym__logical__", "lasym"), False))

    def close(self):
        self.ds.close()

    def _get(self, *names):
        for name in names:
            if name in self.ds.variables:
                return np.asarray(self.ds[name].values)
            if name in self.ds.attrs:
                return self.ds.attrs[name]
        return None

    def _modes(self, nyquist=False):
        suffix = "_nyq" if nyquist else ""
        xm = self._get(f"xm{suffix}")
        xn = self._get(f"xn{suffix}")
        if xm is None or xn is None:
            raise KeyError(f"Missing xm{suffix}/xn{suffix} Fourier mode arrays")
        return np.ravel(xm), np.ravel(xn)

    def _radial_modes(self, name: str, s: float, half=False) -> np.ndarray:
        arr = np.asarray(self._get(name), dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2-D radial-by-mode array")
        # xarray preserves file order; detect whether mode or radius is first.
        if arr.shape[0] not in (self.ns, self.ns - 1) and arr.shape[1] in (
            self.ns,
            self.ns - 1,
        ):
            arr = arr.T
        return interp_radial(arr, s, half=half, full_size=self.ns)

    def _profile_array(self, aliases: tuple[str, ...]):
        for name in aliases:
            arr = self._get(name)
            if arr is not None:
                return name, np.ravel(np.asarray(arr, dtype=float))
        return None, None

    def _profile_grid(self, length: int, half: bool):
        if not half:
            return radial_grid(length)
        if length == self.ns - 1:
            return (np.arange(length) + 0.5) / (self.ns - 1)
        return radial_grid(length, half=True)

    def surface(self, s=1.0, ntheta=128, nphi=128) -> Surface:
        theta_1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi_1d = np.linspace(0, 2 * np.pi / self.nfp, nphi, endpoint=False)
        theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")
        xm, xn = self._modes(False)
        rmnc = self._radial_modes("rmnc", s)
        zmns = self._radial_modes("zmns", s)
        R = fourier_cos(rmnc, xm, xn, theta, phi)
        Z = fourier_sin(zmns, xm, xn, theta, phi)
        if self.lasym and self._get("rmns") is not None:
            R += fourier_sin(self._radial_modes("rmns", s), xm, xn, theta, phi)
        if self.lasym and self._get("zmnc") is not None:
            Z += fourier_cos(self._radial_modes("zmnc", s), xm, xn, theta, phi)
        return Surface(float(s), theta, phi, R, Z)

    def section(self, s=1.0, phi=0.0, ntheta=361) -> Surface:
        """Reconstruct a closed R-Z curve at an exact toroidal angle."""
        theta_1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)
        theta = theta_1d[:, None]
        phi_grid = np.full_like(theta, float(phi))
        xm, xn = self._modes(False)
        R = fourier_cos(self._radial_modes("rmnc", s), xm, xn, theta, phi_grid)
        Z = fourier_sin(self._radial_modes("zmns", s), xm, xn, theta, phi_grid)
        if self.lasym and self._get("rmns") is not None:
            R += fourier_sin(self._radial_modes("rmns", s), xm, xn, theta, phi_grid)
        if self.lasym and self._get("zmnc") is not None:
            Z += fourier_cos(self._radial_modes("zmnc", s), xm, xn, theta, phi_grid)
        R[-1, 0] = R[0, 0]
        Z[-1, 0] = Z[0, 0]
        return Surface(float(s), theta, phi_grid, R, Z)

    def field_map(self, s=1.0, ntheta=128, nzeta=128, coordinates="native"):
        if coordinates.lower() not in {"native", "vmec"}:
            raise ValueError(
                "A wout file contains VMEC angles, not Boozer angles. "
                "Pass a boozmn file to plot a Boozer map."
            )
        theta_1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta_1d = np.linspace(0, 2 * np.pi / self.nfp, nzeta, endpoint=False)
        theta, zeta = np.meshgrid(theta_1d, zeta_1d, indexing="ij")
        xm, xn = self._modes(True)
        bmnc = self._radial_modes("bmnc", s, half=True)
        B = fourier_cos(bmnc, xm, xn, theta, zeta)
        if self.lasym and self._get("bmns") is not None:
            B += fourier_sin(self._radial_modes("bmns", s, half=True), xm, xn, theta, zeta)
        return FieldMap(float(s), theta, zeta, B, coordinates="VMEC")

    def to_data(self) -> EquilibriumData:
        get = self._get
        scalars = {}
        scalar_map = {
            "aspect": ("aspect",),
            "volume_m3": ("volume_p", "volume"),
            "R_major_m": ("Rmajor_p",),
            "A_minor_m": ("Aminor_p",),
            "B_axis_T": ("b0",),
            "beta_total": ("betatotal", "betatot", "beta_vol"),
            "beta_poloidal": ("betapol",),
            "beta_toroidal": ("betator",),
            "toroidal_flux_Wb": ("phi",),
            "plasma_current_A": ("ctor",),
            "ier_flag": ("ier_flag",),
            "fsqr": ("fsqr",),
            "fsqz": ("fsqz",),
            "fsql": ("fsql",),
        }
        for out_name, aliases in scalar_map.items():
            val = get(*aliases)
            if val is not None:
                if out_name == "toroidal_flux_Wb" and np.asarray(val).size > 1:
                    val = np.ravel(val)[-1]
                scalars[out_name] = as_scalar(val)

        profiles = {}
        for name, aliases in PROFILE_ALIASES.items():
            vmec_name, arr = self._profile_array(aliases)
            if arr is None:
                continue
            half = vmec_name in {"iotas", "pres", "mass", "jcurv", "jcuru", "vp"}
            s = self._profile_grid(len(arr), half)
            valid = s >= 0
            profiles[name] = (s[valid], arr[valid])

        iota = profiles.get("iota")
        if iota is not None:
            scalars.update(
                iota_axis=float(iota[1][0]),
                iota_edge=float(iota[1][-1]),
                iota_mean=float(np.nanmean(iota[1])),
                mean_shear=float(np.polyfit(iota[0], iota[1], 1)[0]),
            )

        vp = get("vp")
        if vp is not None:
            v = np.ravel(np.asarray(vp, dtype=float))
            v_for_well = v[1:] if len(v) == self.ns else v
            v0, v1 = edge_extrapolate(v_for_well)
            scalars["magnetic_well"] = (v0 - v1) / v0 if v0 != 0 else np.nan
            s_v = self._profile_grid(len(v_for_well), half=True)
            valid = s_v >= 0
            profiles["magnetic_well"] = (
                np.concatenate(([0.0], s_v[valid], [1.0])),
                np.concatenate(
                    (
                        [0.0],
                        (v0 - v_for_well[valid]) / v0
                        if v0 != 0
                        else np.full(np.count_nonzero(valid), np.nan),
                        [(v0 - v1) / v0 if v0 != 0 else np.nan],
                    )
                ),
            )
        elif get("gmnc") is not None:
            gmnc = np.asarray(get("gmnc"), dtype=float)
            if gmnc.shape[0] != self.ns and gmnc.shape[1] == self.ns:
                gmnc = gmnc.T
            dVds = 4 * np.pi**2 * np.abs(gmnc[1:, 0])
            v0, v1 = edge_extrapolate(dVds)
            scalars["magnetic_well"] = (v0 - v1) / v0 if v0 != 0 else np.nan
            s_v = self._profile_grid(len(dVds), half=True)
            valid = s_v >= 0
            profiles["magnetic_well"] = (
                np.concatenate(([0.0], s_v[valid], [1.0])),
                np.concatenate(
                    (
                        [0.0],
                        (v0 - dVds[valid]) / v0
                        if v0 != 0
                        else np.full(np.count_nonzero(valid), np.nan),
                        [(v0 - v1) / v0 if v0 != 0 else np.nan],
                    )
                ),
            )

        stability = {}
        for out_name, vmec_names in STABILITY_NAMES.items():
            arr = get(*vmec_names)
            if arr is None:
                continue
            arr = np.ravel(np.asarray(arr, dtype=float))
            s = self._profile_grid(len(arr), half=True)
            valid = s >= 0
            stability[out_name] = (s[valid], arr[valid])
        if "D_Mercier" in stability:
            s, dmerc = stability["D_Mercier"]
            mask = (s >= 0.05) & np.isfinite(dmerc)
            if np.any(mask):
                scalars["D_Mercier_min_s>=0.05"] = float(np.min(dmerc[mask]))
                scalars["D_Mercier_negative_fraction"] = float(np.mean(dmerc[mask] < 0))

        label = self.path.stem.replace("wout_", "")
        warnings = []
        if scalars.get("ier_flag", 0) not in (0, None):
            warnings.append(f"VMEC ier_flag={scalars['ier_flag']}; verify convergence.")
        return EquilibriumData(
            source=self.path,
            backend="VMEC",
            label=label,
            nfp=self.nfp,
            scalars=scalars,
            profiles=profiles,
            profile_units={
                "pressure": "Pa",
                "toroidal_current": "A/m²",
                "dV_ds": "m³",
                "magnetic_well": "dimensionless",
            },
            stability=stability,
            metadata={
                "lasym": self.lasym,
                "ns": self.ns,
                "mpol": int(as_scalar(get("mpol"), -1)),
                "ntor": int(as_scalar(get("ntor"), -1)),
                "mgrid_file": as_scalar(get("mgrid_file"), ""),
            },
            warnings=warnings,
            adapter=self,
        )


class BoozerAdapter:
    """Minimal reader for BOOZ_XFORM boozmn NetCDF files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds = xr.open_dataset(self.path, decode_cf=False, mask_and_scale=False)
        self.nfp = int(as_scalar(self._get("nfp_b", "nfp"), 1))
        self.s_grid = self._read_surface_grid()

    def close(self):
        self.ds.close()

    def _get(self, *names):
        for name in names:
            if name in self.ds:
                return np.asarray(self.ds[name].values)
        return None

    def _read_surface_grid(self):
        bmnc = np.asarray(self._get("bmnc_b"))
        xm = np.ravel(self._get("ixm_b", "xm_b"))
        if bmnc.ndim == 2:
            radial_size = bmnc.shape[0] if bmnc.shape[-1] == len(xm) else bmnc.shape[1]
        else:
            radial_size = 1
        s_b = self._get("s_b")
        if s_b is not None and np.asarray(s_b).size == radial_size:
            return np.ravel(np.asarray(s_b, dtype=float))
        jlist = self._get("jlist")
        ns = int(as_scalar(self._get("ns_b", "ns"), 0))
        if jlist is not None and ns > 1:
            labels = np.ravel(np.asarray(jlist, dtype=float))
            if labels.size == radial_size:
                # booz_xform stores half-grid input surface k as jlist=k+2.
                # Its normalized toroidal flux is therefore (jlist-1.5)/(ns-1).
                return (labels - 1.5) / (ns - 1)
        phi_b = self._get("phi_b")
        if phi_b is not None and np.asarray(phi_b).size == radial_size:
            phi_b = np.ravel(np.asarray(phi_b, dtype=float))
            edge = np.nanmax(np.abs(phi_b))
            if edge > 0:
                return np.abs(phi_b) / edge
        return np.linspace(0, 1, radial_size)

    def available_surfaces(self):
        return np.asarray(self.s_grid, dtype=float)

    def surface_labels(self):
        """Return BOOZ_XFORM ``jlist`` labels when available."""
        labels = self._get("jlist")
        if labels is None:
            return None
        labels = np.ravel(np.asarray(labels, dtype=int))
        return labels if labels.size == len(self.s_grid) else None

    def iota_at(self, s: float):
        """Return rotational transform on the nearest packed Boozer surface."""
        iota = self._get("iota_b", "iota")
        if iota is None:
            raise KeyError("boozmn file is missing iota_b")
        values = np.ravel(np.asarray(iota, dtype=float))
        nearest = int(np.argmin(np.abs(self.s_grid - float(s))))
        labels = self.surface_labels()
        if values.size == len(self.s_grid):
            return float(values[nearest])
        if labels is not None:
            # booz_xform documents compute_surfs = jlist - 2, while iota_b
            # retains the full input radial array.
            index = int(labels[nearest]) - 2
            if 0 <= index < values.size:
                return float(values[index])
        grid = np.linspace(0, 1, values.size)
        return float(np.interp(float(self.s_grid[nearest]), grid, values))

    def modes_at(self, s: float):
        """Return ``(m, n, bmnc, bmns)`` Fourier data at normalized flux ``s``."""
        xm = np.ravel(self._get("ixm_b", "xm_b"))
        xn = np.ravel(self._get("ixn_b", "xn_b"))
        bmnc = np.asarray(self._get("bmnc_b"), dtype=float)
        if bmnc.shape[-1] != len(xm):
            bmnc = bmnc.T
        cosine = self._interp_boozer_modes(bmnc, s)
        raw_sine = self._get("bmns_b")
        if raw_sine is None:
            sine = np.zeros_like(cosine)
        else:
            raw_sine = np.asarray(raw_sine, dtype=float)
            if raw_sine.shape[-1] != len(xm):
                raw_sine = raw_sine.T
            sine = self._interp_boozer_modes(raw_sine, s)
        return xm, xn, cosine, sine

    def field_map(self, s=1.0, ntheta=128, nzeta=128, coordinates="boozer"):
        theta_1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta_1d = np.linspace(0, 2 * np.pi / self.nfp, nzeta, endpoint=False)
        theta, zeta = np.meshgrid(theta_1d, zeta_1d, indexing="ij")
        xm, xn, bmnc, bmns = self.modes_at(s)
        B = fourier_cos(bmnc, xm, xn, theta, zeta)
        B += fourier_sin(bmns, xm, xn, theta, zeta)
        return FieldMap(float(s), theta, zeta, B, coordinates="Boozer")

    def _interp_boozer_modes(self, values, s):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.s_grid):
            if values.shape[1] == len(self.s_grid):
                values = values.T
            else:
                return interp_radial(values, s)
        order = np.argsort(self.s_grid)
        grid = np.asarray(self.s_grid)[order]
        values = values[order]
        flat = values.reshape(len(grid), -1)
        out = np.array([np.interp(float(s), grid, flat[:, j]) for j in range(flat.shape[1])])
        return out.reshape(values.shape[1:])
