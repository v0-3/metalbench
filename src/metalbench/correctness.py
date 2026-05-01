from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from metalbench import env
from metalbench.import_utils import (
    import_module_from_path,
    require_generated_contract,
    require_reference_contract,
)


class CorrectnessResult(BaseModel):
    ok: bool
    trials: int
    passed: int
    failed: int
    errors: list[str]


def tree_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(tree_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [tree_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: tree_to_device(value, device) for key, value in obj.items()}
    return obj


def tree_to_cpu(obj: Any) -> Any:
    return tree_to_device(obj, torch.device("cpu"))


def assert_close_tree(actual: Any, expected: Any, *, rtol: float, atol: float) -> None:
    _assert_close_tree(actual, expected, rtol=rtol, atol=atol, path="root")


def run_correctness_trials(
    ref_path: Path,
    kernel_path: Path,
    trials: int = 5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    require_mps: bool = True,
) -> CorrectnessResult:
    if not env.is_mps_available():
        if require_mps:
            env.require_mps()
        return CorrectnessResult(
            ok=False,
            trials=0,
            passed=0,
            failed=0,
            errors=["MPS is not available; correctness trials were skipped."],
        )

    reference_module = import_module_from_path(ref_path, "metalbench_reference")
    generated_module = import_module_from_path(kernel_path, "metalbench_generated")
    require_reference_contract(reference_module)
    require_generated_contract(generated_module)

    mps_device = torch.device("mps")
    init_inputs = _as_args(reference_module.get_init_inputs())
    reference_model = reference_module.Model(*tree_to_cpu(init_inputs))
    generated_model = generated_module.ModelNew(*tree_to_device(init_inputs, mps_device))
    reference_model.eval()
    generated_model.to(mps_device).eval()

    passed = 0
    errors: list[str] = []
    with torch.no_grad():
        for trial_index in range(trials):
            inputs = _as_args(reference_module.get_inputs())
            expected = reference_model(*tree_to_cpu(inputs))
            actual = generated_model(*tree_to_device(inputs, mps_device))
            try:
                assert_close_tree(tree_to_cpu(actual), tree_to_cpu(expected), rtol=rtol, atol=atol)
            except AssertionError as exc:
                errors.append(f"trial {trial_index + 1}: {exc}")
            else:
                passed += 1

    failed = trials - passed
    return CorrectnessResult(
        ok=failed == 0,
        trials=trials,
        passed=passed,
        failed=failed,
        errors=errors,
    )


def _assert_close_tree(actual: Any, expected: Any, *, rtol: float, atol: float, path: str) -> None:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise AssertionError(f"{path}: {exc}") from exc
        return

    if isinstance(actual, tuple) and isinstance(expected, tuple):
        _assert_sequence_close(actual, expected, rtol=rtol, atol=atol, path=path)
        return

    if isinstance(actual, list) and isinstance(expected, list):
        _assert_sequence_close(actual, expected, rtol=rtol, atol=atol, path=path)
        return

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        _assert_mapping_close(actual, expected, rtol=rtol, atol=atol, path=path)
        return

    if actual != expected:
        raise AssertionError(f"{path}: values differ: {actual!r} != {expected!r}")


def _assert_sequence_close(
    actual: tuple[Any, ...] | list[Any],
    expected: tuple[Any, ...] | list[Any],
    *,
    rtol: float,
    atol: float,
    path: str,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{path}: lengths differ: {len(actual)} != {len(expected)}")
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
        _assert_close_tree(
            actual_item,
            expected_item,
            rtol=rtol,
            atol=atol,
            path=f"{path}[{index}]",
        )


def _assert_mapping_close(
    actual: Mapping[Any, Any],
    expected: Mapping[Any, Any],
    *,
    rtol: float,
    atol: float,
    path: str,
) -> None:
    if actual.keys() != expected.keys():
        raise AssertionError(f"{path}: keys differ: {actual.keys()} != {expected.keys()}")
    for key in actual:
        _assert_close_tree(
            actual[key],
            expected[key],
            rtol=rtol,
            atol=atol,
            path=f"{path}[{key!r}]",
        )


def _as_args(obj: Any) -> tuple[Any, ...]:
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, list):
        return tuple(obj)
    return (obj,)


__all__ = [
    "CorrectnessResult",
    "assert_close_tree",
    "run_correctness_trials",
    "tree_to_cpu",
    "tree_to_device",
]
