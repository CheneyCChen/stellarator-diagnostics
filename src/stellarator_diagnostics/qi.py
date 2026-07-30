"""Goodman-compatible squash-stretch-shuffle quasi-isodynamic residual."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from .plots import STYLE, _minor_ticks, _save
from .vmec import BoozerAdapter


@dataclass
class GoodmanQiSurface:
    """Detailed result on one normalized-toroidal-flux surface."""

    s: float
    surface_label: int | None
    iota: float
    residual: float
    squash_residual: float
    stretch_residual: float
    shuffle_residual: float
    b_min: float
    b_max: float
    zeta_offset: float
    alpha: np.ndarray = field(repr=False)
    phi: np.ndarray = field(repr=False)
    original: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)

    def row(self):
        goodman_mirror_ratio = (self.b_max - self.b_min) / (
            self.b_max + self.b_min
        )
        return {
            "s": self.s,
            "surface_label": self.surface_label,
            "iota": self.iota,
            "f_QI": self.residual,
            "squash_residual": self.squash_residual,
            "stretch_residual": self.stretch_residual,
            "shuffle_residual": self.shuffle_residual,
            "B_min_T": self.b_min,
            "B_max_T": self.b_max,
            "mirror_ratio": goodman_mirror_ratio,
            "relative_well_depth": (self.b_max - self.b_min) / self.b_min,
            "zeta_offset_rad": self.zeta_offset,
        }


@dataclass
class GoodmanQiResult:
    """Goodman QI residuals and the fields used to construct them."""

    source: Path
    data: pd.DataFrame
    surfaces: list[GoodmanQiSurface] = field(repr=False)
    nalpha: int
    nphi: int
    nlevels: int

    def summary(self):
        values = self.data["f_QI"].to_numpy(float)
        finite = values[np.isfinite(values)]
        return {
            "source": str(self.source),
            "definition": (
                "Goodman-compatible squash-stretch-shuffle f_QI "
                "with feasible-center projection"
            ),
            "surface_count": len(self.data),
            "nalpha": self.nalpha,
            "nphi": self.nphi,
            "nlevels": self.nlevels,
            "f_QI_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "f_QI_max": float(np.max(finite)) if finite.size else float("nan"),
        }


def _periodic_normalized_integral(values, alpha, phi, scale):
    """Return the domain average divided by ``scale**2``."""
    periodic = np.concatenate((values, values[:1]), axis=0)
    alpha_closed = np.concatenate((alpha, [2 * np.pi]))
    integral_phi = trapezoid(periodic, phi, axis=1)
    integral = trapezoid(integral_phi, alpha_closed)
    domain = 2 * np.pi * (phi[-1] - phi[0])
    return float(integral / (domain * scale**2))


def _squash_well(values, minimum_index):
    """Flatten nonmonotonic points using Goodman et al.'s boundary-to-minimum loops."""
    left = np.minimum.accumulate(values[: minimum_index + 1])
    right = np.minimum.accumulate(values[minimum_index:][::-1])[::-1]
    return np.concatenate((left[:-1], right))


def _inverse_branch(field, position, levels):
    """Invert a monotone branch, averaging positions on flat segments."""
    order = np.argsort(field, kind="stable")
    field = np.asarray(field)[order]
    position = np.asarray(position)[order]
    unique, inverse = np.unique(field, return_inverse=True)
    positions = np.zeros_like(unique, dtype=float)
    counts = np.zeros_like(unique, dtype=float)
    np.add.at(positions, inverse, position)
    np.add.at(counts, inverse, 1.0)
    positions /= counts
    return np.interp(levels, unique, positions)


def _feasible_centers(raw, delta, period):
    """Project branch centers while preserving nesting and the period endpoints."""
    count = len(delta)
    local_low = 0.5 * delta
    local_high = period - 0.5 * delta
    low = local_low.copy()
    high = local_high.copy()
    low[-1] = high[-1] = 0.5 * period
    for index in range(count - 2, -1, -1):
        step = 0.5 * max(delta[index + 1] - delta[index], 0.0)
        low[index] = max(low[index], low[index + 1] - step)
        high[index] = min(high[index], high[index + 1] + step)
    centers = np.empty(count)
    centers[0] = np.clip(raw[0], low[0], high[0])
    for index in range(1, count):
        step = 0.5 * max(delta[index] - delta[index - 1], 0.0)
        lower = max(low[index], centers[index - 1] - step)
        upper = min(high[index], centers[index - 1] + step)
        centers[index] = np.clip(raw[index], lower, upper)
    return centers


def _evaluate_fourier_field(m, n, bmnc, bmns, theta, zeta, chunk_size=128):
    """Evaluate Boozer |B| without allocating modes × alpha × phi in full."""
    theta, zeta = np.broadcast_arrays(
        np.asarray(theta, dtype=float),
        np.asarray(zeta, dtype=float),
    )
    result = np.zeros(theta.shape, dtype=float)
    for start in range(0, len(m), chunk_size):
        stop = min(start + chunk_size, len(m))
        expand = (slice(start, stop),) + (None,) * theta.ndim
        phase = m[expand] * theta[None, ...] - n[expand] * zeta[None, ...]
        result += np.sum(
            bmnc[expand] * np.cos(phase) + bmns[expand] * np.sin(phase),
            axis=0,
        )
    return result


def _auto_zeta_offset(adapter, s, nalpha, nphi):
    """Place the field-period boundaries on the strongest common high-B plane."""
    period = 2 * np.pi / adapter.nfp
    count = max(4 * (nphi - 1), 256)
    zeta = np.linspace(0, period, count, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, max(nalpha, 64), endpoint=False)
    m, n, bmnc, bmns = adapter.modes_at(s)
    field = _evaluate_fourier_field(
        m,
        n,
        bmnc,
        bmns,
        theta[:, None],
        zeta[None, :],
    )
    # In a QI field B_max closes poloidally, so a valid boundary should be
    # high for every poloidal angle rather than only high on average.
    score = np.min(field, axis=0)
    return float(zeta[int(np.argmax(score))])


def _fieldline_wells(adapter, s, nalpha, nphi, zeta_offset):
    alpha = np.linspace(0, 2 * np.pi, nalpha, endpoint=False)
    period = 2 * np.pi / adapter.nfp
    phi = np.linspace(0, period, nphi)
    iota = adapter.iota_at(s)
    m, n, bmnc, bmns = adapter.modes_at(s)
    theta = alpha[:, None] + iota * phi[None, :]
    values = _evaluate_fourier_field(
        m,
        n,
        bmnc,
        bmns,
        theta,
        zeta_offset + phi[None, :],
    )
    return alpha, phi, iota, values


def _compute_surface(
    adapter,
    s,
    surface_label,
    nalpha,
    nphi,
    nlevels,
    zeta_offset,
):
    if zeta_offset is None:
        offset = _auto_zeta_offset(adapter, s, nalpha, nphi)
    else:
        offset = float(zeta_offset) % (2 * np.pi / adapter.nfp)
    alpha, phi, iota, original = _fieldline_wells(
        adapter, s, nalpha, nphi, offset
    )
    period = phi[-1]
    b_min = float(np.min(original))
    b_max = float(np.max(original))
    scale = b_max - b_min
    if not np.isfinite(scale) or scale <= np.finfo(float).eps * max(abs(b_max), 1.0):
        raise ValueError(f"B is constant on s={s:.8g}; Goodman QI normalization is undefined")

    squashed = np.empty_like(original)
    stretched = np.empty_like(original)
    minima = np.argmin(original, axis=1)
    if np.any((minima == 0) | (minima == nphi - 1)):
        count = int(np.count_nonzero((minima == 0) | (minima == nphi - 1)))
        raise ValueError(
            f"{count}/{nalpha} field-line wells on s={s:.8g} have their minimum at a "
            "field-period boundary even after selecting the high-B toroidal plane at "
            f"zeta={offset:.8g} rad. Try a denser Boozer transform or set --zeta-offset "
            "explicitly after inspecting the Boozer contours"
        )

    for index, minimum in enumerate(minima):
        row = _squash_well(original[index], minimum)
        squashed[index] = row
        local_min = row[minimum]
        left_depth = row[0] - local_min
        right_depth = row[-1] - local_min
        if min(left_depth, right_depth) <= np.finfo(float).eps * max(abs(b_max), 1.0):
            raise ValueError(
                f"Field-line well {index} on s={s:.8g} has a degenerate branch"
            )
        stretched[index, : minimum + 1] = b_min + scale * (
            row[: minimum + 1] - local_min
        ) / left_depth
        stretched[index, minimum:] = b_min + scale * (
            row[minimum:] - local_min
        ) / right_depth

    mismatch = trapezoid((original - stretched) ** 2, phi, axis=1)
    floor = np.finfo(float).eps * period * scale**2
    weights = 1.0 / np.maximum(mismatch, floor)
    weights /= np.sum(weights)
    levels = np.linspace(b_min, b_max, nlevels)
    left_positions = np.empty((nalpha, nlevels))
    right_positions = np.empty((nalpha, nlevels))
    for index, minimum in enumerate(minima):
        left_positions[index] = _inverse_branch(
            stretched[index, : minimum + 1], phi[: minimum + 1], levels
        )
        right_positions[index] = _inverse_branch(
            stretched[index, minimum:], phi[minimum:], levels
        )

    branch_widths = right_positions - left_positions
    delta = np.sum(weights[:, None] * branch_widths, axis=0)
    delta[0] = 0.0
    delta[-1] = period
    delta = np.maximum.accumulate(np.clip(delta, 0.0, period))

    width_spread = np.max(np.abs(branch_widths - delta[None, :]))
    if width_spread <= 64 * np.finfo(float).eps * max(period, 1.0):
        # The stretched field already has alpha-independent bounce distance.
        # Preserve it exactly instead of introducing inverse-grid interpolation error.
        target = stretched.copy()
    else:
        target = np.empty_like(original)
        for index in range(nalpha):
            raw_center = 0.5 * (left_positions[index] + right_positions[index])
            center = _feasible_centers(raw_center, delta, period)
            left = center - 0.5 * delta
            right = center + 0.5 * delta
            minimum_position = center[0]
            left_mask = phi <= minimum_position
            target[index, left_mask] = np.interp(
                phi[left_mask], left[::-1], levels[::-1]
            )
            target[index, ~left_mask] = np.interp(phi[~left_mask], right, levels)

    return GoodmanQiSurface(
        s=float(s),
        surface_label=surface_label,
        iota=float(iota),
        residual=_periodic_normalized_integral(
            (original - target) ** 2, alpha, phi, scale
        ),
        squash_residual=_periodic_normalized_integral(
            (original - squashed) ** 2, alpha, phi, scale
        ),
        stretch_residual=_periodic_normalized_integral(
            (squashed - stretched) ** 2, alpha, phi, scale
        ),
        shuffle_residual=_periodic_normalized_integral(
            (stretched - target) ** 2, alpha, phi, scale
        ),
        b_min=b_min,
        b_max=b_max,
        zeta_offset=offset,
        alpha=alpha,
        phi=phi,
        original=original,
        target=target,
    )


def compute_goodman_qi(
    boozmn: str | Path,
    surfaces=None,
    nalpha: int = 64,
    nphi: int = 129,
    nlevels: int = 129,
    zeta_offset: float | None = None,
):
    """Compute a Goodman-compatible normalized QI residual on Boozer surfaces.

    ``surfaces`` are normalized-toroidal-flux values. Each is mapped to the
    nearest surface actually packed into the supplied ``boozmn`` file. The
    shuffle step uses a feasible-center projection that enforces nested,
    equal-bounce-width branches.
    """
    if nalpha < 4 or nphi < 9 or nlevels < 9:
        raise ValueError("Goodman QI grids require nalpha>=4, nphi>=9, and nlevels>=9")
    if nphi % 2 == 0:
        raise ValueError("nphi must be odd so the center of a field period is sampled")
    source = Path(boozmn)
    adapter = BoozerAdapter(source)
    try:
        available = adapter.available_surfaces()
        requested = available if surfaces is None else np.asarray(surfaces, dtype=float)
        indices = []
        for value in requested:
            if not 0 <= value <= 1:
                raise ValueError(f"QI surface s={value} is outside [0, 1]")
            nearest = int(np.argmin(np.abs(available - value)))
            if nearest not in indices:
                indices.append(nearest)
        labels = adapter.surface_labels()
        results = [
            _compute_surface(
                adapter,
                available[index],
                None if labels is None else int(labels[index]),
                nalpha,
                nphi,
                nlevels,
                zeta_offset,
            )
            for index in indices
        ]
    finally:
        adapter.close()
    frame = pd.DataFrame([result.row() for result in results])
    return GoodmanQiResult(source, frame, results, nalpha, nphi, nlevels)


def _plot_profile(result, path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.semilogy(result.data["s"], result.data["f_QI"], "o-", color="black", lw=1.6)
        ax.set_xlabel(r"Normalized toroidal flux $s$")
        ax.set_ylabel(r"Goodman-compatible $f_{\rm QI}$")
        ax.set_title("Quasi-isodynamic residual")
        _minor_ticks(ax)
        return _save(fig, path)


def _plot_wells(surface, path):
    count = min(8, len(surface.alpha))
    indices = np.linspace(0, len(surface.alpha) - 1, count, dtype=int)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, count))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for color, index in zip(colors, indices):
            label = rf"$\alpha/2\pi={surface.alpha[index] / (2 * np.pi):.2f}$"
            axes[0].plot(surface.phi, surface.original[index], color=color, lw=1.2, label=label)
            axes[1].plot(surface.phi, surface.target[index], color=color, lw=1.2)
        axes[0].set_title("Original Boozer field-line wells")
        axes[1].set_title("Goodman-compatible QI target")
        for ax in axes:
            ax.set_xlabel(r"Boozer toroidal angle $\varphi$ [rad]")
            _minor_ticks(ax)
        axes[0].set_ylabel(r"$|B|$ [T]")
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(rf"$s={surface.s:.4f}$, $f_{{\rm QI}}={surface.residual:.3e}$")
        return _save(fig, path)


def diagnose_goodman_qi(
    boozmn: str | Path,
    outdir: str | Path = "qi_diagnostics",
    surfaces=None,
    nalpha: int = 64,
    nphi: int = 129,
    nlevels: int = 129,
    zeta_offset: float | None = None,
):
    """Compute the residual and write CSV, JSON, and diagnostic figures."""
    result = compute_goodman_qi(
        boozmn, surfaces, nalpha, nphi, nlevels, zeta_offset
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "goodman_qi_residual.csv"
    json_path = outdir / "goodman_qi_summary.json"
    result.data.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(result.summary(), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    outputs = [csv_path, json_path, _plot_profile(result, outdir / "goodman_qi_residual.png")]
    for surface in result.surfaces:
        tag = f"{surface.s:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        outputs.append(_plot_wells(surface, outdir / f"goodman_qi_wells_s{tag}.png"))
    return result, outputs
