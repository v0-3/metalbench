from pathlib import Path

import pytest

import metalbench.env
from metalbench.timing import (
    TimingResult,
    _summarize_timings_ms,
    time_generated_metal_kernel,
    time_reference_mps_baseline,
)


def test_summarize_timings_ms_returns_summary_metrics() -> None:
    result = _summarize_timings_ms([1.0, 2.0, 4.0, 8.0], warmup=3, errors=[])

    assert result == TimingResult(
        ok=True,
        device="mps",
        median_ms=3.0,
        mean_ms=3.75,
        min_ms=1.0,
        max_ms=8.0,
        p25_ms=1.75,
        p75_ms=5.0,
        warmup=3,
        trials=4,
        errors=[],
    )


def test_summarize_timings_ms_returns_empty_result_without_samples() -> None:
    result = _summarize_timings_ms([], warmup=2, errors=["no samples"])

    assert result.ok is False
    assert result.device == "mps"
    assert result.median_ms is None
    assert result.mean_ms is None
    assert result.min_ms is None
    assert result.max_ms is None
    assert result.p25_ms is None
    assert result.p75_ms is None
    assert result.warmup == 2
    assert result.trials == 0
    assert result.errors


def test_time_generated_metal_kernel_returns_skipped_result_when_mps_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    generated_path = tmp_path / "generated.py"
    reference_path.write_text("", encoding="utf-8")
    generated_path.write_text("", encoding="utf-8")

    result = time_generated_metal_kernel(reference_path, generated_path, require_mps=False)

    assert result.ok is False
    assert result.device == "mps"
    assert result.trials == 0
    assert any("MPS" in error for error in result.errors)


def test_time_generated_metal_kernel_requires_mps_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    generated_path = tmp_path / "generated.py"
    reference_path.write_text("", encoding="utf-8")
    generated_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="MPS"):
        time_generated_metal_kernel(reference_path, generated_path)


def test_time_reference_mps_baseline_returns_skipped_result_when_mps_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    reference_path.write_text("", encoding="utf-8")

    result = time_reference_mps_baseline(reference_path, require_mps=False)

    assert result.ok is False
    assert result.device == "mps"
    assert result.trials == 0
    assert any("MPS" in error for error in result.errors)


def test_time_reference_mps_baseline_requires_mps_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)
    reference_path = tmp_path / "reference.py"
    reference_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="MPS"):
        time_reference_mps_baseline(reference_path)


@pytest.mark.mps
def test_times_matching_relu_modules_on_mps(tmp_path: Path) -> None:
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

_METAL_SRC = '''
#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    const device float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < n) {
        y[index] = max(x[index], 0.0f);
    }
}
'''

_LIB = torch.mps.compile_shader(_METAL_SRC)


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        _LIB.relu_f32(x, y, x.numel())
        return y
""",
        encoding="utf-8",
    )

    generated_result = time_generated_metal_kernel(
        reference_path,
        generated_path,
        warmup=1,
        trials=2,
    )
    reference_result = time_reference_mps_baseline(reference_path, warmup=1, trials=2)

    for result in (generated_result, reference_result):
        assert result.ok is True
        assert result.device == "mps"
        assert result.median_ms is not None and result.median_ms > 0.0
        assert result.mean_ms is not None and result.mean_ms > 0.0
        assert result.min_ms is not None and result.min_ms > 0.0
        assert result.max_ms is not None and result.max_ms > 0.0
        assert result.warmup == 1
        assert result.trials == 2
        assert result.errors == []
