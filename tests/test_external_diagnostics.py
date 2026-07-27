from pathlib import Path

import numpy as np

from stellarator_diagnostics.external import (
    diagnose_cobra,
    diagnose_dkes,
    diagnose_neo,
    read_cobra,
    read_dkes,
    read_neo,
)


def test_neo_basic_format_and_plot(tmp_path: Path):
    source = tmp_path / "neo_out.case"
    source.write_text(
        "2 1.0D-04 0.2 0.70 2.0 5.0\n"
        "3 8.0D-05 0.3 0.72 2.0 5.0\n"
        "4 6.0D-05 0.4 0.74 2.0 5.0\n",
        encoding="utf-8",
    )
    result = read_neo(source)
    assert result.eout_swi == 1
    assert np.isclose(result.data.loc[0, "epsilon_eff"], (1e-4) ** (2 / 3))
    diagnosed, outputs = diagnose_neo(source, tmp_path / "neo")
    assert diagnosed.summary()["surface_count"] == 3
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_dkes_bounds_and_plots(tmp_path: Path):
    source = tmp_path / "results.case"
    rows = ["DKES results", "19 numeric columns"]
    for efield in (0.0, 0.1):
        for cmul in (1e-4, 1e-3, 1e-2):
            base = 1.0 / (1.0 + 100 * cmul)
            values = [
                cmul,
                efield,
                0,
                0,
                0.95 * base,
                1.05 * base,
                -0.12 * base,
                -0.08 * base,
                0.45 * base,
                0.55 * base,
                1,
                1,
                1,
                1e-8,
                1,
                1,
                1,
                1,
                1,
            ]
            rows.append(" ".join(f"{value:.9e}" for value in values))
    source.write_text("\n".join(rows), encoding="utf-8")
    result = read_dkes(source)
    assert len(result.data) == 6
    assert np.isclose(result.data.iloc[0]["L11"], 1 / (1 + 100e-4))
    diagnosed, outputs = diagnose_dkes(source, tmp_path / "dkes")
    assert diagnosed.summary()["efield_count"] == 2
    assert len(outputs) == 2
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_cobra_new_and_legacy_formats(tmp_path: Path):
    modern = tmp_path / "cobra_grate.modern"
    modern.write_text(
        "0.0 0.0 3\n"
        "2 0.2 -0.10\n"
        "3 0.4 -0.20\n"
        "4 0.6 0.05\n"
        "1.57 0.0 3\n"
        "2 0.2 -0.08\n"
        "3 0.4 -0.22\n"
        "4 0.6 0.03\n",
        encoding="utf-8",
    )
    result = read_cobra(modern)
    assert result.normalized_s
    assert len(result.data) == 6
    assert np.isclose(result.summary()["minimum_eigenvalue"], -0.22)
    diagnosed, outputs = diagnose_cobra(modern, tmp_path / "cobra")
    assert diagnosed.summary()["field_line_count"] == 2
    assert outputs[0].stat().st_size > 1000

    legacy = tmp_path / "cobra_grate.legacy"
    legacy.write_text(
        "0.0 0.0 2\n"
        "2 -0.10\n"
        "3 0.02\n",
        encoding="utf-8",
    )
    old = read_cobra(legacy)
    assert not old.normalized_s
    assert list(old.data["s"]) == [2.0, 3.0]
