from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_pyproject() -> dict[str, Any]:
    with Path("pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_pyproject_declares_required_dependencies() -> None:
    pyproject = load_pyproject()
    dependencies = pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["dependency-groups"]["dev"]
    hf_dependencies = pyproject["project"]["optional-dependencies"]["hf"]

    for package in ["torch", "numpy", "pandas", "pydantic>=2", "typer", "rich"]:
        assert any(dependency.startswith(package) for dependency in dependencies)

    for package in ["pytest", "ruff", "mypy"]:
        assert any(dependency.startswith(package) for dependency in dev_dependencies)

    assert any(dependency.startswith("datasets") for dependency in hf_dependencies)


def test_pyproject_declares_console_script_and_tools() -> None:
    pyproject = load_pyproject()

    assert pyproject["project"]["scripts"]["metalbench"] == "metalbench.cli:app"
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/metalbench"
    ]
    assert "mps: tests requiring Apple Metal MPS runtime" in pyproject["tool"]["pytest"][
        "ini_options"
    ]["markers"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py310"
    assert pyproject["tool"]["mypy"]["strict"] is True
