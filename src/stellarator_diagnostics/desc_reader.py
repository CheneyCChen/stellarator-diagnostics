"""Optional DESC adapter; DESC is imported only when needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import EquilibriumData, FieldMap, Surface


class DescAdapter:
    def __init__(self, path: str | Path, family_index: int = -1):
        self.path = Path(path)
        try:
            from desc.io import load
        except ImportError as exc:
            raise ImportError(
                "DESC support requires `pip install 'stellarator-diagnostics[desc]'` "
                "or installation inside an existing DESC environment."
            ) from exc
        obj = load(str(self.path))
        if hasattr(obj, "__len__") and not hasattr(obj, "compute"):
            obj = obj[family_index]
        self.eq = obj
        self.nfp = int(obj.NFP)

    @staticmethod
    def _scalar(data, name, default=np.nan):
        if name not in data:
            return default
        arr = np.asarray(data[name], dtype=float)
        return float(np.nanmean(arr))

    def surface(self, s=1.0, ntheta=128, nphi=128):
        from desc.grid import LinearGrid

        grid = LinearGrid(rho=float(np.sqrt(s)), M=ntheta // 2, N=nphi // 2, NFP=self.nfp)
        data = self.eq.compute(["R", "Z", "phi"], grid=grid)
        shape = (grid.num_theta, grid.num_zeta)
        return Surface(
            float(s),
            np.asarray(grid.nodes[:, 1]).reshape(shape),
            np.asarray(data["phi"]).reshape(shape),
            np.asarray(data["R"]).reshape(shape),
            np.asarray(data["Z"]).reshape(shape),
        )

    def field_map(self, s=1.0, ntheta=128, nzeta=128, coordinates="native"):
        from desc.grid import LinearGrid

        if coordinates.lower() not in {"native", "desc"}:
            raise ValueError(
                "Use DESC's `plot_boozer_surface` for a Boozer-coordinate map; "
                "the unified adapter currently returns native DESC coordinates."
            )
        grid = LinearGrid(rho=float(np.sqrt(s)), M=ntheta // 2, N=nzeta // 2, NFP=self.nfp)
        data = self.eq.compute("|B|", grid=grid)
        shape = (grid.num_theta, grid.num_zeta)
        return FieldMap(
            float(s),
            np.asarray(grid.nodes[:, 1]).reshape(shape),
            np.asarray(grid.nodes[:, 2]).reshape(shape),
            np.asarray(data["|B|"]).reshape(shape),
            coordinates="DESC",
        )

    def _profile(self, name: str, nrho=101):
        from desc.grid import LinearGrid

        grid = LinearGrid(rho=np.linspace(0, 1, nrho), M=0, N=0, NFP=self.nfp)
        data = self.eq.compute(name, grid=grid)
        return grid.nodes[:, 0] ** 2, np.asarray(data[name], dtype=float)

    def to_data(self):
        scalar_names = [
            "R0",
            "a",
            "A",
            "V",
            "<|B|>_vol",
            "<beta>_vol",
            "<|F|>_vol",
            "W",
        ]
        available = {}
        for name in scalar_names:
            try:
                available.update(self.eq.compute(name))
            except Exception:
                continue
        scalars = {
            "aspect": self._scalar(available, "A"),
            "volume_m3": self._scalar(available, "V"),
            "R_major_m": self._scalar(available, "R0"),
            "A_minor_m": self._scalar(available, "a"),
            "B_mean_T": self._scalar(available, "<|B|>_vol"),
            "beta_total": self._scalar(available, "<beta>_vol"),
            "force_error_mean": self._scalar(available, "<|F|>_vol"),
            "energy_J": self._scalar(available, "W"),
        }
        scalars = {k: v for k, v in scalars.items() if np.isfinite(v)}
        profiles = {}
        for out_name, desc_name in {
            "iota": "iota",
            "pressure": "p",
            "shear": "shear",
            "dV_ds": "V_r(r)",
            "current": "current",
        }.items():
            try:
                profiles[out_name] = self._profile(desc_name)
            except Exception:
                continue
        if "iota" in profiles:
            s, iota = profiles["iota"]
            scalars.update(
                iota_axis=float(iota[0]),
                iota_edge=float(iota[-1]),
                iota_mean=float(np.trapz(iota, s)),
                mean_shear=float(np.polyfit(s, iota, 1)[0]),
            )
        stability = {}
        for out_name, desc_name in {
            "D_Mercier": "D_Mercier",
            "magnetic_well": "magnetic well",
        }.items():
            try:
                stability[out_name] = self._profile(desc_name)
            except Exception:
                continue
        if "D_Mercier" in stability:
            s, dm = stability["D_Mercier"]
            mask = s >= 0.05
            scalars["D_Mercier_min_s>=0.05"] = float(np.nanmin(dm[mask]))
            scalars["D_Mercier_negative_fraction"] = float(np.mean(dm[mask] < 0))
        return EquilibriumData(
            source=self.path,
            backend="DESC",
            label=self.path.stem,
            nfp=self.nfp,
            scalars=scalars,
            profiles=profiles,
            profile_units={"pressure": "Pa", "current": "A"},
            stability=stability,
            metadata={
                "L": int(self.eq.L),
                "M": int(self.eq.M),
                "N": int(self.eq.N),
                "sym": bool(self.eq.sym),
            },
            adapter=self,
        )
