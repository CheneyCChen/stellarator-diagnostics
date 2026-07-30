"""Readers and publication-style plots for STELLOPT transport/stability outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import STYLE, _minor_ticks, _save


@dataclass
class NeoResult:
    source: Path
    data: pd.DataFrame
    eout_swi: int

    def summary(self):
        values = self.data["epstot"].to_numpy(float)
        eps = self.data["epsilon_eff"].to_numpy(float)
        return {
            "source": str(self.source),
            "eout_swi": self.eout_swi,
            "surface_count": len(self.data),
            "epstot_min": _finite_reduce(values, np.nanmin),
            "epstot_max": _finite_reduce(values, np.nanmax),
            "epsilon_eff_max": _finite_reduce(eps, np.nanmax),
        }


@dataclass
class DkesResult:
    source: Path
    data: pd.DataFrame

    def summary(self):
        result = {
            "source": str(self.source),
            "run_count": len(self.data),
            "cmul_count": int(self.data["cmul"].nunique()),
            "efield_count": int(self.data["efield"].nunique()),
        }
        if "surface_index" in self.data:
            result["surface_count"] = int(self.data["surface_index"].nunique())
        for coefficient in ("L11", "L31", "L33"):
            spread = self.data[f"{coefficient}_relative_spread"].to_numpy(float)
            result[f"{coefficient}_median_relative_spread"] = _finite_reduce(spread, np.nanmedian)
            result[f"{coefficient}_max_relative_spread"] = _finite_reduce(spread, np.nanmax)
        if "D11_star" in self.data:
            values = self.data["D11_star"].to_numpy(float)
            result["D11_star_min"] = _finite_reduce(values, np.nanmin)
            result["D11_star_max"] = _finite_reduce(values, np.nanmax)
        return result


@dataclass
class CobraResult:
    source: Path
    data: pd.DataFrame
    normalized_s: bool

    def summary(self):
        valid = self.data[
            ~self.data["solver_failed"] & np.isfinite(self.data["growth_rate"])
        ]
        rates = valid["growth_rate"].to_numpy(float)
        if len(valid):
            worst_index = int(np.nanargmax(rates))
            worst = valid.iloc[worst_index]
            maximum = float(worst["growth_rate"])
            worst_s = float(worst["s"])
            worst_zeta = float(worst["zeta0"])
            worst_theta = float(worst["theta0"])
        else:
            maximum = worst_s = worst_zeta = worst_theta = float("nan")
        return {
            "source": str(self.source),
            "field_line_count": int(self.data.groupby(["zeta0", "theta0"]).ngroups),
            "surface_count": int(self.data["s"].nunique()),
            "normalized_s": self.normalized_s,
            "valid_point_count": len(valid),
            "failure_count": int(self.data["solver_failed"].sum()),
            "maximum_growth_rate": maximum,
            "unstable_fraction": float(np.mean(rates > 0)) if rates.size else float("nan"),
            "worst_s": worst_s,
            "worst_zeta0": worst_zeta,
            "worst_theta0": worst_theta,
            "sign_convention": (
                "positive COBRAVMEC signed growth rate is ideal-ballooning unstable; "
                "negative is stable"
            ),
        }


NEO_BASIC = ["surface_label", "epstot", "reff", "iota", "b_ref", "r_ref"]
NEO_DETAILED = NEO_BASIC + [
    "epspar1",
    "epspar2",
    "ctrone",
    "ctrtot",
    "bareph",
    "barept",
    "yps",
]
DKES_COLUMNS = [
    "cmul",
    "efield",
    "weov",
    "wtov",
    "L11m",
    "L11p",
    "L31m",
    "L31p",
    "L33m",
    "L33p",
    "scal11",
    "scal13",
    "scal33",
    "rsds_max",
    "chip",
    "psip",
    "btheta",
    "bzeta",
    "vp",
]


def _numeric_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace("D", "E").replace("d", "e").replace(",", " ").split()
        if not parts:
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return rows


def read_neo(path: str | Path):
    """Read STELLOPT NEO ``neo_out`` formats EOUT_SWI=1, 2, or 10."""
    path = Path(path)
    rows = _numeric_rows(path)
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent NEO row widths: {sorted(widths)}")
    width = widths.pop()
    if width >= 13:
        frame = pd.DataFrame([row[:13] for row in rows], columns=NEO_DETAILED)
        eout_swi = 2
    elif width >= 6:
        frame = pd.DataFrame([row[:6] for row in rows], columns=NEO_BASIC)
        eout_swi = 1
    elif width == 3:
        frame = pd.DataFrame(rows, columns=["b_ref", "r_ref", "epstot"])
        frame.insert(0, "surface_label", np.arange(1, len(frame) + 1, dtype=float))
        eout_swi = 10
    else:
        raise ValueError(f"Unsupported NEO output width {width}; expected 3, 6, or 13")
    frame["epsilon_eff"] = np.where(
        frame["epstot"] >= 0,
        np.power(frame["epstot"], 2.0 / 3.0),
        np.nan,
    )
    return NeoResult(path, frame, eout_swi)


def read_dkes(path: str | Path):
    """Read the 19-column STELLOPT DKES ``results`` table."""
    path = Path(path)
    rows = _numeric_rows(path)
    valid = [row for row in rows if len(row) >= 10]
    if not valid:
        raise ValueError("DKES results require at least the first 10 numeric columns")
    width = min(max(len(row) for row in valid), len(DKES_COLUMNS))
    if any(len(row) < width for row in valid):
        width = min(len(row) for row in valid)
    frame = pd.DataFrame([row[:width] for row in valid], columns=DKES_COLUMNS[:width])
    for coefficient in ("L11", "L31", "L33"):
        lower = frame[f"{coefficient}m"]
        upper = frame[f"{coefficient}p"]
        middle = 0.5 * (lower + upper)
        scale = np.maximum(np.abs(middle), np.finfo(float).tiny)
        frame[coefficient] = middle
        frame[f"{coefficient}_relative_spread"] = np.abs(upper - lower) / scale
    # STELLOPT calls the normalized monoenergetic matrix element L11;
    # stellarator transport literature commonly denotes the same normalized
    # radial coefficient D11*.  scal11 converts it to the code's 1-keV H+
    # MKS reference coefficient, before any Maxwellian energy convolution.
    frame["D11_star_m"] = frame["L11m"]
    frame["D11_star_p"] = frame["L11p"]
    frame["D11_star"] = frame["L11"]
    if "scal11" in frame:
        frame["D11_ref_m2_s_m"] = frame["L11m"] * frame["scal11"]
        frame["D11_ref_m2_s_p"] = frame["L11p"] * frame["scal11"]
        frame["D11_ref_m2_s"] = frame["L11"] * frame["scal11"]
    return DkesResult(path, frame)


def read_cobra(path: str | Path):
    """Read old two-column and newer three-column COBRA ``cobra_grate`` blocks."""
    path = Path(path)
    numeric = _numeric_rows(path)
    records = []
    cursor = 0
    normalized_s = True
    while cursor < len(numeric):
        header = numeric[cursor]
        cursor += 1
        if len(header) < 3:
            raise ValueError(f"Invalid COBRA block header at numeric row {cursor}")
        zeta0, theta0 = header[:2]
        surface_count = round(header[2])
        if surface_count <= 0 or cursor + surface_count > len(numeric):
            raise ValueError(f"Invalid COBRA surface count {surface_count}")
        for row in numeric[cursor : cursor + surface_count]:
            if len(row) >= 3:
                surface_index, s, growth_rate = row[:3]
            elif len(row) == 2:
                surface_index, growth_rate = row
                s = surface_index
                normalized_s = False
            else:
                raise ValueError("COBRA surface row requires two or three columns")
            records.append(
                {
                    "zeta0": zeta0,
                    "theta0": theta0,
                    "surface_index": round(surface_index),
                    "s": s,
                    "growth_rate": growth_rate,
                    # COBRAVMEC sets the full rate vector to 100 before returning
                    # when its ballooning solve fails.  It is not a physical rate.
                    "solver_failed": bool(
                        np.isclose(growth_rate, 100.0, rtol=0.0, atol=1e-8)
                    ),
                }
            )
        cursor += surface_count
    if not records:
        raise ValueError("No COBRA field-line blocks found")
    frame = pd.DataFrame.from_records(records)
    return CobraResult(path, frame, normalized_s)


def plot_neo(result: NeoResult, path: str | Path):
    """Plot effective-ripple profiles without implying an unavailable flux mapping."""
    frame = result.data.sort_values("surface_label")
    x = frame["surface_label"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
        axes[0].plot(x, frame["epstot"], "o-", color="#1f4e79", lw=1.7, ms=3.5)
        axes[0].set_ylabel(r"$\epsilon_{\rm eff}^{3/2}$")
        axes[0].set_yscale("log")
        axes[0].set_title(r"NEO effective ripple ($1/\nu$ regime)")
        axes[1].plot(x, frame["epsilon_eff"], "o-", color="#b13c2e", lw=1.7, ms=3.5)
        axes[1].set_ylabel(r"$\epsilon_{\rm eff}$")
        axes[1].set_xlabel("NEO surface label")
        for ax in axes:
            ax.tick_params(which="both", direction="in")
        return _save(fig, path)


def plot_dkes(result: DkesResult, outdir: str | Path):
    """Plot DKES variational bounds and their relative spread."""
    outdir = Path(outdir)
    frame = result.data.sort_values(["efield", "cmul"])
    efields = sorted(frame["efield"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, max(len(efields), 2)))
    coefficient_path = outdir / "dkes_coefficients.png"
    convergence_path = outdir / "dkes_convergence.png"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.0), sharex=True)
        for ax, coefficient in zip(axes, ("L11", "L31", "L33")):
            for color, efield in zip(colors, efields):
                group = frame[frame["efield"] == efield]
                x = group["cmul"].to_numpy(float)
                lower = group[f"{coefficient}m"].to_numpy(float)
                upper = group[f"{coefficient}p"].to_numpy(float)
                middle = group[coefficient].to_numpy(float)
                order = np.argsort(x)
                x, lower, upper, middle = (
                    value[order] for value in (x, lower, upper, middle)
                )
                label = rf"$E/v={efield:.3g}$"
                ax.plot(x, middle, "o-", color=color, lw=1.5, ms=3, label=label)
                ax.fill_between(x, np.minimum(lower, upper), np.maximum(lower, upper),
                                color=color, alpha=0.17, linewidth=0)
            ax.set_ylabel(coefficient)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=_symlog_threshold(frame[coefficient]))
            ax.tick_params(which="both", direction="in")
        axes[0].set_title("DKES monoenergetic coefficients and variational bounds")
        axes[0].legend(ncols=2, fontsize=8)
        axes[-1].set_xlabel(r"Collisionality parameter $\nu/v$ [m$^{-1}$]")
        _save(fig, coefficient_path)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.3), sharex=True)
        for ax, coefficient in zip(axes, ("L11", "L31", "L33")):
            for color, efield in zip(colors, efields):
                group = frame[frame["efield"] == efield].sort_values("cmul")
                ax.plot(
                    group["cmul"],
                    100 * group[f"{coefficient}_relative_spread"],
                    "o-",
                    color=color,
                    lw=1.4,
                    ms=3,
                    label=rf"$E/v={efield:.3g}$",
                )
            ax.axhline(3.0, color="0.25", ls="--", lw=0.9)
            ax.set_ylabel(f"{coefficient} spread [%]")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(which="both", direction="in")
        axes[0].set_title("DKES variational-bound convergence")
        axes[0].legend(ncols=2, fontsize=8)
        axes[-1].set_xlabel(r"Collisionality parameter $\nu/v$ [m$^{-1}$]")
        _save(fig, convergence_path)
    return [coefficient_path, convergence_path]


def plot_dkes_d11_scan(result: DkesResult, outdir: str | Path):
    """Plot the standard monoenergetic radial coefficient across surfaces."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = result.data.sort_values(["surface_index", "efield", "cmul"])
    surfaces = sorted(frame["surface_index"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, max(len(surfaces), 2)))
    coefficient_path = outdir / "dkes_D11_star_scan.png"
    convergence_path = outdir / "dkes_D11_convergence.png"

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.1))
        for color, surface in zip(colors, surfaces):
            surface_frame = frame[frame["surface_index"] == surface]
            for efield, group in surface_frame.groupby("efield"):
                group = group.sort_values("cmul")
                label = rf"$j={int(surface)},\ E_s/v={efield:.3g}$"
                ax.plot(
                    group["cmul"],
                    group["D11_star"],
                    "o-",
                    color=color,
                    lw=1.5,
                    ms=3.2,
                    label=label,
                )
                ax.fill_between(
                    group["cmul"],
                    np.minimum(group["D11_star_m"], group["D11_star_p"]),
                    np.maximum(group["D11_star_m"], group["D11_star_p"]),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        ax.set_xscale("log")
        if np.all(frame["D11_star"] > 0):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=_symlog_threshold(frame["D11_star"]))
        ax.set_xlabel(r"Collisionality parameter $\nu/v$ [m$^{-1}$]")
        ax.set_ylabel(
            r"Normalized monoenergetic radial coefficient $D_{11}^*$ "
            r"[m$^{-1}$ T$^{-2}$]"
        )
        ax.set_title(r"DKES radial transport: $D_{11}^*$")
        ax.legend(fontsize=7, ncols=2)
        _minor_ticks(ax)
        _save(fig, coefficient_path)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for color, surface in zip(colors, surfaces):
            surface_frame = frame[frame["surface_index"] == surface]
            for efield, group in surface_frame.groupby("efield"):
                group = group.sort_values("cmul")
                ax.plot(
                    group["cmul"],
                    100 * group["L11_relative_spread"],
                    "o-",
                    color=color,
                    lw=1.4,
                    ms=3,
                    label=rf"$j={int(surface)},\ E_s/v={efield:.3g}$",
                )
        ax.axhline(3.0, color="0.25", ls="--", lw=0.9, label="3% guide")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Collisionality parameter $\nu/v$ [m$^{-1}$]")
        ax.set_ylabel(r"$D_{11}^*$ variational spread [%]")
        ax.set_title("DKES radial-coefficient convergence")
        ax.legend(fontsize=7, ncols=2)
        _minor_ticks(ax)
        _save(fig, convergence_path)

    return [coefficient_path, convergence_path]


def plot_cobra(result: CobraResult, path: str | Path):
    """Plot COBRAVMEC signed growth rates; positive values are unstable."""
    frame = result.data.sort_values(["zeta0", "theta0", "s"])
    valid = frame[~frame["solver_failed"] & np.isfinite(frame["growth_rate"])]
    failed = frame[frame["solver_failed"]]
    grouped = valid.groupby("s")["growth_rate"]
    envelope = grouped.agg(["min", "median", "max"]).reset_index().sort_values("s")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for _, group in valid.groupby(["zeta0", "theta0"]):
            ax.plot(group["s"], group["growth_rate"], color="0.65", lw=0.7, alpha=0.45)
        if not envelope.empty:
            ax.fill_between(
                envelope["s"],
                envelope["min"],
                envelope["max"],
                color="#4c78a8",
                alpha=0.15,
                linewidth=0,
                label="field-line envelope",
            )
            ax.plot(envelope["s"], envelope["max"], color="#b13c2e", lw=2, label="maximum")
            ax.plot(
                envelope["s"],
                envelope["median"],
                color="#1f4e79",
                lw=1.5,
                label="median",
            )
        if not failed.empty:
            ax.scatter(
                failed["s"],
                np.zeros(len(failed)),
                marker="x",
                color="#7f0000",
                s=34,
                label="solver failure",
                zorder=5,
            )
        ax.axhline(0, color="black", lw=0.9)
        if len(valid):
            upper = max(float(np.nanmax(valid["growth_rate"])), 0.0)
            if upper > 0:
                ax.axhspan(0, upper, color="#d73027", alpha=0.07, zorder=-10)
        else:
            ax.text(
                0.5,
                0.5,
                "No valid COBRAVMEC points",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel(
            r"Normalized toroidal flux $s$" if result.normalized_s else "VMEC surface index"
        )
        ax.set_ylabel(r"COBRAVMEC signed growth rate $\gamma$")
        ax.set_title("Ideal ballooning scan (positive = unstable)")
        ax.legend(ncols=3, fontsize=8)
        _minor_ticks(ax)
        return _save(fig, path)


def diagnose_neo(
    source: str | Path,
    outdir: str | Path,
    surface_labels: list[int] | None = None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = read_neo(source)
    if surface_labels is not None:
        if len(surface_labels) != len(result.data):
            raise ValueError(
                "NEO output row count does not match the prepared boozmn surface mapping: "
                f"{len(result.data)} != {len(surface_labels)}"
            )
        result.data.insert(
            0,
            "boozmn_surface_position",
            result.data["surface_label"].to_numpy(copy=True),
        )
        result.data["surface_label"] = np.asarray(surface_labels, dtype=int)
    result.data.to_csv(outdir / "neo_normalized.csv", index=False)
    figure = plot_neo(result, outdir / "neo_effective_ripple.png")
    return result, [figure]


def diagnose_dkes(source: str | Path, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = read_dkes(source)
    result.data.to_csv(outdir / "dkes_normalized.csv", index=False)
    return result, plot_dkes(result, outdir)


def diagnose_cobra(source: str | Path, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = read_cobra(source)
    result.data.to_csv(outdir / "cobra_normalized.csv", index=False)
    figure = plot_cobra(result, outdir / "cobra_ballooning.png")
    return result, [figure]


def _symlog_threshold(values):
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return max(float(np.nanpercentile(finite, 10)) if finite.size else 1e-12, 1e-30)


def _finite_reduce(values, reducer):
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(reducer(values))
