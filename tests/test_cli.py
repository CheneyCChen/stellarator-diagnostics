import pytest

from stellarator_diagnostics.cli import _parser


def test_cli_reports_installed_version(capsys):
    with pytest.raises(SystemExit, match="0"):
        _parser().parse_args(["--version"])
    assert capsys.readouterr().out.strip() == "stell-diag 0.7.0"
