from pathlib import Path

import pytest
import torch

import metalbench.env
from metalbench.correctness import (
    assert_close_tree,
    run_correctness_trials,
    tree_to_cpu,
    tree_to_device,
)


def test_tree_to_device_moves_tensors_in_nested_containers() -> None:
    tensor = torch.tensor([1.0, 2.0])
    nested = {
        "tensor": tensor,
        "tuple": (tensor + 1, "kept"),
        "list": [tensor + 2],
        "dict": {"inner": tensor + 3},
    }

    moved = tree_to_device(nested, torch.device("cpu"))

    assert isinstance(moved, dict)
    assert moved["tensor"].device.type == "cpu"
    assert moved["tuple"][0].device.type == "cpu"
    assert moved["tuple"][1] == "kept"
    assert moved["list"][0].device.type == "cpu"
    assert moved["dict"]["inner"].device.type == "cpu"


def test_tree_to_cpu_moves_tensors_to_cpu() -> None:
    tensor = torch.tensor([1.0])

    moved = tree_to_cpu((tensor,))

    assert isinstance(moved, tuple)
    assert moved[0].device.type == "cpu"


def test_assert_close_tree_accepts_matching_nested_trees() -> None:
    expected = {
        "a": torch.tensor([1.0, 2.0]),
        "b": [torch.tensor([3.0]), (torch.tensor([4.0]),)],
    }
    actual = {
        "a": torch.tensor([1.0, 2.0]),
        "b": [torch.tensor([3.0]), (torch.tensor([4.0]),)],
    }

    assert_close_tree(actual, expected, rtol=1e-4, atol=1e-4)


def test_assert_close_tree_reports_tensor_mismatches_with_path() -> None:
    actual = {"output": [torch.tensor([1.0])]}
    expected = {"output": [torch.tensor([2.0])]}

    with pytest.raises(AssertionError) as exc_info:
        assert_close_tree(actual, expected, rtol=1e-4, atol=1e-4)

    message = str(exc_info.value)
    assert "root['output'][0]" in message
    assert "Tensor-likes are not close" in message


def test_run_correctness_trials_returns_skipped_result_when_mps_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    generated_path = tmp_path / "generated.py"
    reference_path.write_text("", encoding="utf-8")
    generated_path.write_text("", encoding="utf-8")

    result = run_correctness_trials(
        reference_path,
        generated_path,
        require_mps=False,
    )

    assert result.ok is False
    assert result.trials == 0
    assert result.passed == 0
    assert result.failed == 0
    assert any("MPS" in error for error in result.errors)


def test_run_correctness_trials_requires_mps_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    generated_path = tmp_path / "generated.py"
    reference_path.write_text("", encoding="utf-8")
    generated_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="MPS"):
        run_correctness_trials(reference_path, generated_path)


@pytest.mark.mps
def test_run_correctness_trials_accepts_matching_relu_modules(tmp_path: Path) -> None:
    if not metalbench.env.is_mps_available():
        pytest.skip("MPS is not available")

    reference_path = tmp_path / "reference.py"
    reference_path.write_text(
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_init_inputs():
    return ()


def get_inputs():
    return (torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0]),)
""",
        encoding="utf-8",
    )
    generated_path = tmp_path / "generated.py"
    generated_path.write_text(
        """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
""",
        encoding="utf-8",
    )

    result = run_correctness_trials(reference_path, generated_path, trials=2)

    assert result.ok is True
    assert result.trials == 2
    assert result.passed == 2
    assert result.failed == 0
    assert result.errors == []
