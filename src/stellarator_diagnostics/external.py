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
        for coefficient in ("L11", "L31", "L33"):
            spread = self.data[f"{coefficient}_relative_spread"].to_numpy(float)
            result[f"{coefficient}_median_relative_spread"] = _finite_reduce(spread, np.nanmedian)
            result[f"{coefficient}_max_relative_spread"] = _finite_reduce(spread, np.nanmax)
        return result


@dataclass
class CobraResult:
    source: Path
    data: pd.DataFrame
    normalized_s: bool

    def summary(self):
        rates = self.data["growth_rate"].to_numpy(float)
        if np.any(np.isfinite(rates)):
            worst_index = int(np.nanargmin(rates))
            worst = self.data.iloc[worst_index]
            minimum = float(worst["growth_rate"])
            worst_s = float(worst["s"])
            worst_zeta = float(worst["zeta0"])
            worst_theta = float(worst["theta0"])
        else:
            minimum = worst_s = worst_zeta = worst_theta = float("nan")
        return {
            "source": str(self.source),
            "field_line_count": int(self.data.groupby(["zeta0", "theta0"]).ngroups),
            "surface_count": int(self.data["s"].nunique()),
            "normalized_s": self.normalized_s,
            "minimum_eigenvalue": minimum,
            "unstable_fraction": float(np.mean(rates < 0)),
            "worst_s": worst_s,
            "worst_zeta0": worst_zeta,
            "worst_theta0": worst_theta,
            "sign_convention": "negative eigenvalue is ideal-ballooning unstable",
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


def plot_cobra(result: CobraResult, path: str | Path):
    """Plot COBRA field-line scans; negative eigenvalues are unstable."""
    frame = result.data.sort_values(["zeta0", "theta0", "s"])
    grouped = frame.groupby("s")["growth_rate"]
    envelope = grouped.agg(["min", "median", "max"]).reset_index().sort_values("s")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for _, group in frame.groupby(["zeta0", "theta0"]):
            ax.plot(group["s"], group["growth_rate"], color="0.65", lw=0.7, alpha=0.45)
        ax.fill_between(
            envelope["s"],
            envelope["min"],
            envelope["max"],
            color="#4c78a8",
            alpha=0.15,
            linewidth=0,
            label="field-line envelope",
        )
        ax.plot(envelope["s"], envelope["min"], color="#b13c2e", lw=2, label="minimum")
        ax.plot(envelope["s"], envelope["median"], color="#1f4e79", lw=1.5, label="median")
        ax.axhline(0, color="black", lw=0.9)
        lower = min(float(np.nanmin(frame["growth_rate"])), 0.0)
        if lower < 0:
            ax.axhspan(lower, 0, color="#d73027", alpha=0.07, zorder=-10)
        ax.set_xlabel(
            r"Normalized toroidal flux $s$" if result.normalized_s else "VMEC surface index"
        )
        ax.set_ylabel(r"COBRA eigenvalue $\lambda$")
        ax.set_title("Ideal ballooning scan (negative = unstable)")
        ax.legend(ncols=3, fontsize=8)
        _minor_ticks(ax)
        return _save(fig, path)


def diagnose_neo(source: str | Path, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = read_neo(source)
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
