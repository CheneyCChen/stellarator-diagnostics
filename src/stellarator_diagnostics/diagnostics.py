"""Derived diagnostics that are independent of the equilibrium backend."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import numpy as np


def rational_surfaces(
    s: np.ndarray,
    iota: np.ndarray,
    max_denominator: int = 12,
    min_denominator: int = 2,
) -> list[dict[str, float | int | str]]:
    """Locate low-order rational iota surfaces by piecewise-linear crossings."""
    s = np.asarray(s, dtype=float)
    iota = np.asarray(iota, dtype=float)
    valid = np.isfinite(s) & np.isfinite(iota)
    s, iota = s[valid], iota[valid]
    if len(s) < 2:
        return []
    lo, hi = sorted((float(np.min(iota)), float(np.max(iota))))
    fractions = {
        Fraction(n, d)
        for d in range(min_denominator, max_denominator + 1)
        for n in range(int(np.floor(lo * d)), int(np.ceil(hi * d)) + 1)
        if lo <= n / d <= hi
    }
    found = []
    for frac in sorted(fractions, key=float):
        target = float(frac)
        delta = iota - target
        crossings = np.where(delta[:-1] * delta[1:] <= 0)[0]
        rational_crossings = []
        for idx in crossings:
            if iota[idx + 1] == iota[idx]:
                s_cross = 0.5 * (s[idx] + s[idx + 1])
            else:
                s_cross = s[idx] + (target - iota[idx]) * (s[idx + 1] - s[idx]) / (
                    iota[idx + 1] - iota[idx]
                )
            if any(
                np.isclose(s_cross, previous, rtol=0.0, atol=1e-12)
                for previous in rational_crossings
            ):
                continue
            rational_crossings.append(float(s_cross))
            found.append(
                {
                    "rational": f"{frac.numerator}/{frac.denominator}",
                    "numerator": frac.numerator,
                    "denominator": frac.denominator,
                    "iota": target,
                    "s": float(s_cross),
                }
            )
    return found


def nfp_resonances(
    iota_min: float,
    iota_max: float,
    nfp: int,
    max_reduced_denominator: int = 12,
    max_poloidal_mode: int = 20,
    margin: float = 0.03,
) -> list[dict[str, float | int | str]]:
    """Return low-order resonances compatible with stellarator field periodicity.

    A VMEC harmonic has toroidal mode number ``n = k * NFP``. For each reduced
    rational iota=p/q, this routine finds the lowest equivalent (m,n) pair with
    n divisible by NFP and retains it when m is sufficiently small.
    """
    lo, hi = sorted((float(iota_min), float(iota_max)))
    candidates = set()
    for q in range(2, max_reduced_denominator + 1):
        for p in range(1, q + 1):
            frac = Fraction(p, q)
            value = float(frac)
            if lo - margin <= value <= hi + margin:
                candidates.add(frac)

    result = []
    for frac in sorted(candidates, key=float):
        scale = nfp // gcd(frac.numerator, nfp)
        n_mode = frac.numerator * scale
        m_mode = frac.denominator * scale
        if m_mode > max_poloidal_mode:
            continue
        result.append(
            {
                "rational": f"{frac.numerator}/{frac.denominator}",
                "iota": float(frac),
                "m": m_mode,
                "n": n_mode,
                "in_profile": lo <= float(frac) <= hi,
            }
        )
    return result
