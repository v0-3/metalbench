from typer.testing import CliRunner

from metalbench.cli import app


def test_cli_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "metalbench" in result.output
