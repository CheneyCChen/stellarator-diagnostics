"""Backend-neutral data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(slots=True)
class Surface:
    """A sampled flux surface in cylindrical coordinates."""

    s: float
    theta: Array
    phi: Array
    R: Array
    Z: Array

    @property
    def X(self) -> Array:
        return self.R * np.cos(self.phi)

    @property
    def Y(self) -> Array:
        return self.R * np.sin(self.phi)


@dataclass(slots=True)
class FieldMap:
    """A scalar field sampled on a periodic angular grid."""

    s: float
    theta: Array
    zeta: Array
    values: Array
    name: str = "|B|"
    units: str = "T"
    coordinates: str = "VMEC"


@dataclass
class EquilibriumData:
    """Normalized output shared by VMEC and DESC readers."""

    source: Path
    backend: str
    label: str
    nfp: int
    scalars: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, tuple[Array, Array]] = field(default_factory=dict)
    profile_units: dict[str, str] = field(default_factory=dict)
    stability: dict[str, tuple[Array, Array]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    adapter: Any = field(default=None, repr=False)

    def surface(self, s: float = 1.0, ntheta: int = 128, nphi: int = 128) -> Surface:
        if self.adapter is None:
            raise RuntimeError("No geometry adapter is attached")
        return self.adapter.surface(s=s, ntheta=ntheta, nphi=nphi)

    def section(self, s: float, phi: float, ntheta: int = 361) -> Surface:
        """Return a closed flux-surface section at a toroidal angle."""
        if self.adapter is None:
            raise RuntimeError("No geometry adapter is attached")
        if hasattr(self.adapter, "section"):
            return self.adapter.section(s=s, phi=phi, ntheta=ntheta)
        surface = self.adapter.surface(s=s, ntheta=ntheta, nphi=720)
        index = int(np.argmin(np.abs(surface.phi[0] - phi)))
        return Surface(
            float(s),
            surface.theta[:, index : index + 1],
            surface.phi[:, index : index + 1],
            surface.R[:, index : index + 1],
            surface.Z[:, index : index + 1],
        )

    def field_map(
        self,
        s: float = 1.0,
        ntheta: int = 128,
        nzeta: int = 128,
        coordinates: str = "native",
    ) -> FieldMap:
        if self.adapter is None:
            raise RuntimeError("No field adapter is attached")
        return self.adapter.field_map(s=s, ntheta=ntheta, nzeta=nzeta, coordinates=coordinates)

    def scalar_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend,
            "source": str(self.source),
            "nfp": self.nfp,
            **self.scalars,
        }

    def close(self) -> None:
        """Release backend files held by the attached adapter."""
        if self.adapter is not None and hasattr(self.adapter, "close"):
            self.adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
