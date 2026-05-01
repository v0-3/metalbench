import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType


def import_module_from_path(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Module path not found: {path}")
    if not path.is_file():
        raise ValueError(f"Module path is not a file: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for module {module_name}: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f"Failed to import module {module_name} from path {path}") from exc

    return module


def require_reference_contract(module: ModuleType) -> None:
    invalid_members = [
        name
        for name, predicate in (
            ("Model", inspect.isclass),
            ("get_inputs", callable),
            ("get_init_inputs", callable),
        )
        if not predicate(getattr(module, name, None))
    ]
    if invalid_members:
        members = ", ".join(invalid_members)
        raise ValueError(f"Reference module missing or invalid members: {members}")


def require_generated_contract(module: ModuleType) -> None:
    if not inspect.isclass(getattr(module, "ModelNew", None)):
        raise ValueError("Generated module missing or invalid members: ModelNew")


__all__ = [
    "import_module_from_path",
    "require_generated_contract",
    "require_reference_contract",
]
