import sys
from pathlib import Path
from types import ModuleType

import pytest

from metalbench.import_utils import (
    import_module_from_path,
    require_generated_contract,
    require_reference_contract,
)


def write_module(tmp_path: Path, source: str, name: str = "module.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_import_module_from_path_returns_module_with_loaded_attributes(
    tmp_path: Path,
) -> None:
    path = write_module(tmp_path, "VALUE = 42\n")

    module = import_module_from_path(path, "synthetic_module")

    assert module.__name__ == "synthetic_module"
    assert module.VALUE == 42
    assert sys.modules["synthetic_module"] is module


def test_import_module_from_path_raises_when_path_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError, match="Module path not found"):
        import_module_from_path(path, "missing_module")


def test_import_module_from_path_raises_when_path_is_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Module path is not a file"):
        import_module_from_path(tmp_path, "directory_module")


def test_import_module_from_path_wraps_import_time_failures(tmp_path: Path) -> None:
    path = write_module(tmp_path, "raise RuntimeError('boom')\n")

    with pytest.raises(ImportError, match="Failed to import module broken_module") as exc_info:
        import_module_from_path(path, "broken_module")

    assert str(path) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_require_reference_contract_accepts_valid_module() -> None:
    module = ModuleType("reference")

    class Model:
        pass

    module.__dict__["Model"] = Model
    module.__dict__["get_inputs"] = lambda: ()
    module.__dict__["get_init_inputs"] = lambda: ()

    require_reference_contract(module)


def test_require_reference_contract_raises_for_missing_members() -> None:
    module = ModuleType("reference")

    with pytest.raises(ValueError) as exc_info:
        require_reference_contract(module)

    message = str(exc_info.value)
    assert "Model" in message
    assert "get_inputs" in message
    assert "get_init_inputs" in message


def test_require_reference_contract_raises_for_invalid_member_types() -> None:
    module = ModuleType("reference")
    module.__dict__["Model"] = object()
    module.__dict__["get_inputs"] = 1
    module.__dict__["get_init_inputs"] = None

    with pytest.raises(ValueError) as exc_info:
        require_reference_contract(module)

    message = str(exc_info.value)
    assert "Model" in message
    assert "get_inputs" in message
    assert "get_init_inputs" in message


def test_require_generated_contract_accepts_valid_module() -> None:
    module = ModuleType("generated")

    class ModelNew:
        pass

    module.__dict__["ModelNew"] = ModelNew

    require_generated_contract(module)


def test_require_generated_contract_raises_for_missing_model_new() -> None:
    module = ModuleType("generated")

    with pytest.raises(ValueError, match="ModelNew"):
        require_generated_contract(module)


def test_require_generated_contract_raises_for_invalid_model_new() -> None:
    module = ModuleType("generated")
    module.__dict__["ModelNew"] = object()

    with pytest.raises(ValueError, match="ModelNew"):
        require_generated_contract(module)
