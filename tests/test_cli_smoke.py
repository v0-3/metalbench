import json
from hashlib import sha256
from pathlib import Path

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


def write_problem(kernelbench_dir: Path, source: str = "class Model:\n    pass\n") -> Path:
    level_dir = kernelbench_dir / "level1"
    level_dir.mkdir(parents=True)
    path = level_dir / "19_ReLU.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_cli_kb_list_emits_json_without_source(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "class Model:\n    pass\n"
    path = write_problem(kernelbench_dir, source)

    result = CliRunner().invoke(
        app,
        ["kb", "list", "--kernelbench-dir", str(kernelbench_dir), "--level", "1"],
    )

    assert result.exit_code == 0
    problems = json.loads(result.output)
    assert problems == [
        {
            "level": 1,
            "problem_id": 19,
            "name": "ReLU",
            "path": str(path),
            "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        }
    ]
    assert "source" not in problems[0]


def test_cli_kb_list_missing_directory_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["kb", "list", "--kernelbench-dir", str(tmp_path / "missing"), "--level", "1"],
    )

    assert result.exit_code == 1
    assert "KernelBench directory not found" in result.output


def test_cli_kb_show_emits_json_without_source_by_default(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    path = write_problem(kernelbench_dir)

    result = CliRunner().invoke(
        app,
        [
            "kb",
            "show",
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--level",
            "1",
            "--problem-id",
            "19",
        ],
    )

    assert result.exit_code == 0
    problem = json.loads(result.output)
    assert problem["name"] == "ReLU"
    assert problem["path"] == str(path)
    assert "source" not in problem


def test_cli_kb_show_print_source_includes_source(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "class Model:\n    pass\n"
    write_problem(kernelbench_dir, source)

    result = CliRunner().invoke(
        app,
        [
            "kb",
            "show",
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--level",
            "1",
            "--problem-id",
            "19",
            "--print-source",
        ],
    )

    assert result.exit_code == 0
    problem = json.loads(result.output)
    assert problem["source"] == source
