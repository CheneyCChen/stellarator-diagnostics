"""Static, publication-friendly plots for interactive and batch use."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator

from .diagnostics import nfp_resonances
from .model import EquilibriumData, FieldMap, Surface
from .vmec import BoozerAdapter

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _minor_ticks(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def plot_profiles(eq: EquilibriumData, path: str | Path):
    names = [
        n
        for n in ("pressure", "dV_ds", "magnetic_well", "current", "shear")
        if n in eq.profiles
    ]
    if not names:
        return None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 2.4 * len(names)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            s, value = eq.profiles[name]
            ax.plot(s, value, color="black", lw=1.8)
            unit = eq.profile_units.get(name, "")
            if name == "magnetic_well":
                ax.axhline(0, color="0.5", lw=0.8)
                ax.set_ylabel(r"$[V'(0)-V'(s)]/V'(0)$")
            else:
                ax.set_ylabel(f"{name}" + (f" [{unit}]" if unit else ""))
            _minor_ticks(ax)
        axes[-1].set_xlabel(r"Normalized toroidal flux $s$")
        fig.suptitle("Radial profiles")
        return _save(fig, path)


def plot_iota(eq: EquilibriumData, path: str | Path):
    """Plot iota with low-order resonances allowed by NFP periodicity."""
    if "iota" not in eq.profiles:
        return None
    s, iota = eq.profiles["iota"]
    resonances = nfp_resonances(np.nanmin(iota), np.nanmax(iota), eq.nfp)
    colors = plt.get_cmap("tab10")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(s, iota, color="black", lw=2.0, zorder=5)
        for idx, resonance in enumerate(resonances):
            value = float(resonance["iota"])
            color = colors(idx % 10)
            ax.axhline(value, color=color, ls="--", lw=1.0, alpha=0.85)
            label = (
                rf"${resonance['rational']}$" + rf"  $(m,n)=({resonance['m']},{resonance['n']})$"
            )
            ax.text(
                1.01,
                value,
                label,
                color=color,
                fontsize=8,
                va="center",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
            )
            if resonance["in_profile"]:
                delta = iota - value
                crossings = np.where(delta[:-1] * delta[1:] <= 0)[0]
                for crossing in crossings:
                    pair_iota = iota[crossing : crossing + 2]
                    pair_s = s[crossing : crossing + 2]
                    order = np.argsort(pair_iota)
                    s_cross = np.interp(value, pair_iota[order], pair_s[order])
                    ax.plot(s_cross, value, "o", ms=4, color=color, zorder=6)
                    ax.axvline(s_cross, color=color, ls=":", lw=0.7, alpha=0.55)
        values = [float(item["iota"]) for item in resonances]
        ymin = min([float(np.nanmin(iota)), *values])
        ymax = max([float(np.nanmax(iota)), *values])
        pad = max(0.006, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"Normalized toroidal flux $s$")
        ax.set_ylabel(r"Rotational transform $\iota$")
        ax.set_title(rf"Rotational transform and $N_{{\rm FP}}={eq.nfp}$ resonances")
        _minor_ticks(ax)
        return _save(fig, path)


def plot_mercier_total(eq: EquilibriumData, path: str | Path, analysis_min_s: float = 0.05):
    """Plot total Mercier criterion separately from its decomposition."""
    if "D_Mercier" not in eq.stability:
        return None
    s, value = eq.stability["D_Mercier"]
    mask = s >= analysis_min_s
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.plot(s[mask], value[mask], color="black", lw=1.9, label=r"$D_{\rm Mercier}$")
        ax.axhline(0, color="0.45", lw=0.8)
        ax.set_xlim(analysis_min_s, 1)
        ax.set_ylim(-4, 4)
        ax.set_xlabel(r"Normalized toroidal flux $s$")
        ax.set_ylabel(r"$D_{\rm Mercier}$")
        ax.set_title(rf"Mercier stability criterion ($s\geq{analysis_min_s:g}$)")
        _minor_ticks(ax)
        return _save(fig, path)


def plot_mercier_terms(eq: EquilibriumData, path: str | Path, analysis_min_s: float = 0.05):
    """Plot the four VMEC Mercier contributions without the total."""
    names = ("D_shear", "D_well", "D_current", "D_geodesic")
    available = [name for name in names if name in eq.stability]
    if not available:
        return None
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for name in available:
            s, value = eq.stability[name]
            mask = s >= analysis_min_s
            ax.plot(s[mask], value[mask], lw=1.7, label=name)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlim(analysis_min_s, 1)
        ax.set_ylim(-4, 4)
        ax.set_xlabel(r"Normalized toroidal flux $s$")
        ax.set_ylabel("Mercier contribution")
        ax.set_title(rf"Mercier decomposition ($s\geq{analysis_min_s:g}$)")
        ax.legend(ncols=2, fontsize=8)
        _minor_ticks(ax)
        return _save(fig, path)


def plot_stability(eq: EquilibriumData, path: str | Path, analysis_min_s: float = 0.05):
    """Backward-compatible alias for the total Mercier plot."""
    return plot_mercier_total(eq, path, analysis_min_s=analysis_min_s)


def plot_cross_sections(
    eq: EquilibriumData,
    path: str | Path,
    surfaces=(0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    toroidal_angles=None,
):
    """Plot closed R-Z flux surfaces over half of one field period."""
    if toroidal_angles is None:
        toroidal_angles = (0.0, 1 / 6, 1 / 3, 1 / 2)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
        axes = axes.ravel()
        for ax, frac in zip(axes, toroidal_angles):
            phi = frac * 2 * np.pi / eq.nfp
            for s in surfaces:
                section = eq.section(s=s, phi=phi, ntheta=361)
                r = np.asarray(section.R[:, 0]).copy()
                z = np.asarray(section.Z[:, 0]).copy()
                r[-1], z[-1] = r[0], z[0]
                linewidth = 1.8 if np.isclose(s, 1) else 0.9
                color = "black" if np.isclose(s, 1) else cmap(norm(s))
                ax.plot(r, z, color=color, lw=linewidth)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$R$ [m]")
            ax.set_ylabel(r"$Z$ [m]")
            degrees = frac * 360 / eq.nfp
            fraction = Fraction(frac).limit_denominator(24)
            if fraction.numerator == 0:
                fraction_text = "0"
            elif fraction.denominator == 1:
                fraction_text = str(fraction.numerator)
            else:
                fraction_text = rf"\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
            ax.set_title(rf"$\phi={degrees:g}^\circ" + rf"={fraction_text}(2\pi/N_{{\rm FP}})$")
            _minor_ticks(ax)
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(scalar_map, ax=axes.tolist(), shrink=0.82, pad=0.04)
        cbar.set_label(r"Normalized toroidal flux $s$")
        fig.suptitle(
            rf"Flux surfaces over half a field period "
            rf"($N_{{\rm FP}}={eq.nfp}$)",
            fontsize=12,
        )
        return _save(fig, path)


def plot_boundary_angles(
    eq: EquilibriumData,
    path: str | Path,
    toroidal_angles=None,
):
    """Overlay closed plasma-boundary sections at several toroidal angles."""
    if toroidal_angles is None:
        toroidal_angles = (0.0, 1 / 8, 1 / 4, 3 / 8, 1 / 2)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(toroidal_angles)))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for frac, color in zip(toroidal_angles, colors):
            phi = frac * 2 * np.pi / eq.nfp
            section = eq.section(s=1.0, phi=phi, ntheta=721)
            r = np.asarray(section.R[:, 0], dtype=float).copy()
            z = np.asarray(section.Z[:, 0], dtype=float).copy()
            r[-1], z[-1] = r[0], z[0]
            degrees = frac * 360 / eq.nfp
            ax.plot(r, z, color=color, lw=1.7, label=rf"$\phi={degrees:g}^\circ$")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$ [m]")
        ax.set_ylabel(r"$Z$ [m]")
        ax.set_title(rf"Boundary sections at different toroidal angles ($N_{{\rm FP}}={eq.nfp}$)")
        ax.legend(ncols=2, fontsize=9)
        _minor_ticks(ax)
        return _save(fig, path)


def plot_field_map(field: FieldMap, path: str | Path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        mesh = ax.pcolormesh(
            field.zeta[0] / np.pi,
            field.theta[:, 0] / np.pi,
            field.values,
            shading="auto",
            cmap="viridis",
        )
        fig.colorbar(mesh, ax=ax, label=f"{field.name} [{field.units}]")
        ax.set_xlabel(r"$\zeta/\pi$")
        ax.set_ylabel(r"$\theta/\pi$")
        ax.set_title(rf"{field.coordinates} coordinates, $s={field.s:g}$")
        _minor_ticks(ax)
        return _save(fig, path)


def plot_surface_3d(
    surface: Surface,
    path: str | Path,
    field_values: np.ndarray | None = None,
    nfp: int = 1,
    view: str = "perspective",
):
    """Plot a surface with face color proportional to magnetic-field strength."""
    if nfp > 1:
        period = 2 * np.pi / nfp
        phi = np.concatenate([surface.phi + k * period for k in range(nfp)], axis=1)
        R = np.tile(surface.R, (1, nfp))
        Z = np.tile(surface.Z, (1, nfp))
        surface = Surface(surface.s, np.tile(surface.theta, (1, nfp)), phi, R, Z)
        if field_values is not None:
            field_values = np.tile(field_values, (1, nfp))

    # Matplotlib does not infer periodic connectivity. Duplicate the first
    # toroidal and poloidal samples explicitly so no white seam remains at
    # either angular boundary.
    theta = np.concatenate(
        [surface.theta, surface.theta[:, :1]],
        axis=1,
    )
    phi = np.concatenate(
        [surface.phi, surface.phi[:, :1] + 2 * np.pi],
        axis=1,
    )
    R = np.concatenate([surface.R, surface.R[:, :1]], axis=1)
    Z = np.concatenate([surface.Z, surface.Z[:, :1]], axis=1)
    theta = np.concatenate([theta, theta[:1] + 2 * np.pi], axis=0)
    phi = np.concatenate([phi, phi[:1]], axis=0)
    R = np.concatenate([R, R[:1]], axis=0)
    Z = np.concatenate([Z, Z[:1]], axis=0)
    surface = Surface(surface.s, theta, phi, R, Z)
    if field_values is not None:
        field_values = np.concatenate([field_values, field_values[:, :1]], axis=1)
        field_values = np.concatenate([field_values, field_values[:1]], axis=0)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        if field_values is None:
            facecolors = None
            cmap = "viridis"
        else:
            field_values = np.asarray(field_values, dtype=float)
            norm = Normalize(
                vmin=float(np.nanmin(field_values)),
                vmax=float(np.nanmax(field_values)),
            )
            color_map = plt.get_cmap("plasma")
            facecolors = color_map(norm(field_values))
            cmap = None
        ax.plot_surface(
            surface.X,
            surface.Y,
            surface.Z,
            rstride=max(1, surface.R.shape[0] // 64),
            cstride=max(1, surface.R.shape[1] // 64),
            cmap=cmap,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=field_values is None,
        )
        ax.set_xlabel(r"$X$ [m]")
        ax.set_ylabel(r"$Y$ [m]")
        ax.set_zlabel(r"$Z$ [m]")
        if view == "top":
            ax.view_init(elev=90, azim=-90)
            ax.set_zticks([])
            ax.set_zlabel("")
            ax.zaxis.set_visible(False)
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_title(rf"Top view of flux surface at $s={surface.s:g}$")
        elif view == "perspective":
            ax.view_init(elev=24, azim=-58)
            ax.set_title(rf"Flux surface at $s={surface.s:g}$")
        else:
            raise ValueError("view must be 'perspective' or 'top'")
        z_extent = np.ptp(surface.Z)
        if view == "top":
            z_extent = 0.05 * max(np.ptp(surface.X), np.ptp(surface.Y))
        ax.set_box_aspect((np.ptp(surface.X), np.ptp(surface.Y), z_extent))
        if field_values is not None:
            scalar_map = ScalarMappable(norm=norm, cmap=color_map)
            scalar_map.set_array([])
            cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.66, pad=0.08)
            cbar.set_label(r"$|B|$ [T]")
        return _save(fig, path)


def plot_boozer_surfaces(
    boozmn: str | Path,
    path: str | Path,
    surfaces=(0.25, 0.5, 0.75, 1.0),
    ncontours: int = 18,
):
    """Plot unfilled |B| contours on several Boozer-coordinate surfaces."""
    adapter = BoozerAdapter(boozmn)
    available = adapter.available_surfaces()
    selected = [float(available[np.argmin(np.abs(available - s))]) for s in surfaces]
    fields = [adapter.field_map(s=s, ntheta=160, nzeta=160) for s in selected]
    bmin = min(float(np.nanmin(field.values)) for field in fields)
    bmax = max(float(np.nanmax(field.values)) for field in fields)
    levels = np.linspace(bmin, bmax, ncontours)
    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("viridis")
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8.8, 3.7 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()
        used_mature_plotter = False
        try:
            import booz_xform as bx

            transform = bx.Booz_xform()
            transform.read_boozmn(str(boozmn))
            s_b = np.asarray(transform.s_b, dtype=float)
            if s_b.size == 0:
                raise RuntimeError("BOOZ_XFORM did not read any output surfaces")
            for ax, requested_s in zip(axes_flat, selected):
                plt.sca(ax)
                js = int(np.argmin(np.abs(s_b - requested_s)))
                bx.surfplot(
                    transform,
                    js=js,
                    fill=False,
                    ntheta=160,
                    nphi=160,
                    ncontours=ncontours,
                    cmap="viridis",
                    linewidths=0.8,
                    vmin=bmin,
                    vmax=bmax,
                )
                ax.set_title(rf"$s={s_b[js]:.3f}$")
            used_mature_plotter = True
        except (ImportError, RuntimeError, OSError):
            for ax, selected_s, field in zip(axes_flat, selected, fields):
                ax.contour(
                    field.zeta / np.pi,
                    field.theta / np.pi,
                    field.values,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    linewidths=0.8,
                )
                ax.set_title(rf"$s={selected_s:.3f}$")
                ax.set_xlabel(r"$\zeta_B/\pi$")
                ax.set_ylabel(r"$\theta_B/\pi$")
                _minor_ticks(ax)
        for ax in axes_flat[len(selected) :]:
            ax.set_visible(False)
        scalar_map = ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        cbar = fig.colorbar(scalar_map, ax=axes_flat[: len(selected)].tolist(), shrink=0.86)
        cbar.set_label(r"$|B|$ [T]")
        suffix = "" if used_mature_plotter else " (BOOZ_XFORM reader)"
        fig.suptitle(r"Magnetic-field strength in Boozer coordinates" + suffix)
        return _save(fig, path)


def plot_boozer_surface_files(
    boozmn: str | Path,
    outdir: str | Path,
    surfaces=(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0),
    ncontours: int = 18,
):
    """Write one unfilled BOOZ_XFORM contour figure for each requested s."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adapter = BoozerAdapter(boozmn)
    available = adapter.available_surfaces()
    selected = [float(available[np.argmin(np.abs(available - s))]) for s in surfaces]
    fields = [adapter.field_map(s=s, ntheta=192, nzeta=192) for s in selected]
    bmin = min(float(np.nanmin(field.values)) for field in fields)
    bmax = max(float(np.nanmax(field.values)) for field in fields)
    levels = np.linspace(bmin, bmax, ncontours)
    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("viridis")

    transform = None
    mature_module = None
    try:
        import booz_xform as bx

        transform = bx.Booz_xform()
        transform.read_boozmn(str(boozmn))
        if np.asarray(transform.s_b).size == 0:
            raise RuntimeError("BOOZ_XFORM did not read any output surfaces")
        mature_module = bx
    except (ImportError, RuntimeError, OSError):
        transform = None
        mature_module = None

    outputs = []
    with plt.rc_context(STYLE):
        for selected_s, field in zip(selected, fields):
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            if transform is not None:
                s_b = np.asarray(transform.s_b, dtype=float)
                js = int(np.argmin(np.abs(s_b - selected_s)))
                plt.sca(ax)
                mature_module.surfplot(
                    transform,
                    js=js,
                    fill=False,
                    ntheta=192,
                    nphi=192,
                    ncontours=ncontours,
                    cmap="viridis",
                    linewidths=0.9,
                    vmin=bmin,
                    vmax=bmax,
                )
                actual_s = float(s_b[js])
            else:
                ax.contour(
                    field.zeta / np.pi,
                    field.theta / np.pi,
                    field.values,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    linewidths=0.9,
                )
                ax.set_xlabel(r"$\zeta_B/\pi$")
                ax.set_ylabel(r"$\theta_B/\pi$")
                actual_s = selected_s
                _minor_ticks(ax)
            ax.set_title(rf"$|B|$ in Boozer coordinates at $s={actual_s:.3f}$")
            if len(fig.axes) == 1:
                scalar_map = ScalarMappable(norm=norm, cmap=cmap)
                scalar_map.set_array([])
                cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.9)
                cbar.set_label(r"$|B|$ [T]")
            tag = f"{actual_s:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            output = outdir / f"boozer_s{tag}.png"
            outputs.append(_save(fig, output))
    return outputs


def plot_fieldline_traces(
    eq: EquilibriumData,
    path: str | Path,
    s: float = 0.5,
    alphas=(0.0, 0.5, 1.0, 1.5),
    periods: int = 4,
):
    """Plot |B| along straight-field-line trajectories."""
    if "iota" not in eq.profiles:
        return None
    s_iota, iota_profile = eq.profiles["iota"]
    iota = float(np.interp(s, s_iota, iota_profile))
    field = eq.field_map(s=s, ntheta=192, nzeta=192)
    theta_grid = field.theta[:, 0]
    zeta_grid = field.zeta[0]
    theta_period = 2 * np.pi
    zeta_period = 2 * np.pi / eq.nfp
    theta_closed = np.r_[theta_grid, theta_period]
    zeta_closed = np.r_[zeta_grid, zeta_period]
    values_closed = np.pad(field.values, ((0, 1), (0, 1)), mode="wrap")
    interpolator = RegularGridInterpolator(
        (theta_closed, zeta_closed), values_closed, bounds_error=False
    )
    zeta = np.linspace(0, periods * zeta_period, 1200)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for alpha_pi in alphas:
            theta = alpha_pi * np.pi + iota * zeta
            points = np.column_stack((np.mod(theta, theta_period), np.mod(zeta, zeta_period)))
            ax.plot(
                zeta / zeta_period,
                interpolator(points),
                label=rf"$\alpha/\pi={alpha_pi:g}$",
            )
        ax.set_xlabel("Field periods traversed")
        ax.set_ylabel(r"$|B|$ [T]")
        ax.set_title(rf"Field-line traces at $s={s:g}$, $\iota={iota:.4f}$")
        ax.legend(ncols=2, fontsize=8)
        _minor_ticks(ax)
        return _save(fig, path)


def plot_long_fieldline_trace(
    eq: EquilibriumData,
    path: str | Path,
    s: float = 0.5,
    alpha_pi: float = 0.0,
    periods: int = 200,
):
    """Plot one field line over many field periods."""
    if "iota" not in eq.profiles:
        return None
    s_iota, iota_profile = eq.profiles["iota"]
    iota = float(np.interp(s, s_iota, iota_profile))
    field = eq.field_map(s=s, ntheta=256, nzeta=256)
    theta_grid = field.theta[:, 0]
    zeta_grid = field.zeta[0]
    theta_period = 2 * np.pi
    zeta_period = 2 * np.pi / eq.nfp
    interpolator = RegularGridInterpolator(
        (np.r_[theta_grid, theta_period], np.r_[zeta_grid, zeta_period]),
        np.pad(field.values, ((0, 1), (0, 1)), mode="wrap"),
        bounds_error=False,
    )
    zeta = np.linspace(0, periods * zeta_period, periods * 300 + 1)
    theta = alpha_pi * np.pi + iota * zeta
    points = np.column_stack((np.mod(theta, theta_period), np.mod(zeta, zeta_period)))
    values = interpolator(points)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.plot(zeta / zeta_period, values, color="black", lw=0.75)
        ax.set_xlim(0, periods)
        ax.set_xlabel("Field periods traversed")
        ax.set_ylabel(r"$|B|$ [T]")
        ax.set_title(
            rf"Long field-line trace: $s={s:g}$, "
            rf"$\alpha/\pi={alpha_pi:g}$, $\iota={iota:.4f}$"
        )
        _minor_ticks(ax)
        return _save(fig, path)


def plot_comparison(equilibria: list[EquilibriumData], path: str | Path):
    names = sorted(set.intersection(*(set(eq.profiles) for eq in equilibria)))
    names = [n for n in ("iota", "pressure", "dV_ds", "shear") if n in names]
    if not names:
        return None
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(names), 1, figsize=(7.5, 2.5 * len(names)), sharex=True)
        axes = np.atleast_1d(axes)
        for eq in equilibria:
            for ax, name in zip(axes, names):
                s, value = eq.profiles[name]
                ax.plot(s, value, lw=1.6, label=eq.label)
                ax.set_ylabel(name)
                _minor_ticks(ax)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel(r"Normalized toroidal flux $s$")
        fig.suptitle("Equilibrium comparison")
        return _save(fig, path)
