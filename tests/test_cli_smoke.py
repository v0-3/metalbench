import json

from typer.testing import CliRunner

import metalbench.env
from metalbench.cli import app


def test_cli_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "metalbench" in result.output


def test_cli_env_exits_successfully_and_emits_json() -> None:
    result = CliRunner().invoke(app, ["env"])

    assert result.exit_code == 0
    environment = json.loads(result.output)
    assert environment["device"] == "mps"
    assert "cuda" not in result.output.lower()


def test_cli_env_require_mps_exits_with_error_when_mps_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)

    result = CliRunner().invoke(app, ["env", "--require-mps"])

    assert result.exit_code == 1
    assert "MPS" in result.output
    assert "cuda" not in result.output.lower()
