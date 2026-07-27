"""Numerical helpers with explicit VMEC mesh handling."""

from __future__ import annotations

import numpy as np


def as_scalar(value, default=np.nan):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode(errors="replace").strip()
    return item.item() if hasattr(item, "item") else item


def radial_grid(n: int, half: bool = False) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    if half:
        return (np.arange(n) - 0.5) / (n - 1)
    return np.linspace(0.0, 1.0, n)


def interp_radial(
    values: np.ndarray,
    s: float,
    half: bool = False,
    full_size: int | None = None,
) -> np.ndarray:
    """Interpolate an array whose first dimension is radial."""
    values = np.asarray(values)
    s = float(np.clip(s, 0.0, 1.0))
    if half and full_size is not None and values.shape[0] == full_size - 1:
        grid = (np.arange(values.shape[0]) + 0.5) / (full_size - 1)
    else:
        grid = radial_grid(values.shape[0], half=half)
    valid = (grid >= 0) & np.isfinite(grid)
    grid = grid[valid]
    vals = values[valid]
    if len(grid) == 1:
        return vals[0]
    flat = vals.reshape(len(grid), -1)
    out = np.array([np.interp(s, grid, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape(vals.shape[1:])


def fourier_cos(
    coeff: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    phase = (
        np.asarray(xm)[:, None, None] * theta[None, :, :]
        - np.asarray(xn)[:, None, None] * zeta[None, :, :]
    )
    return np.einsum("m,mij->ij", np.asarray(coeff), np.cos(phase))


def fourier_sin(
    coeff: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    phase = (
        np.asarray(xm)[:, None, None] * theta[None, :, :]
        - np.asarray(xn)[:, None, None] * zeta[None, :, :]
    )
    return np.einsum("m,mij->ij", np.asarray(coeff), np.sin(phase))


def edge_extrapolate(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if len(values) else (np.nan, np.nan)
    return (float(1.5 * values[0] - 0.5 * values[1]), float(1.5 * values[-1] - 0.5 * values[-2]))
