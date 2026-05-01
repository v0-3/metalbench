from pathlib import Path

import pytest

import metalbench.eval_one as eval_one_module
from metalbench.correctness import CorrectnessResult
from metalbench.eval_one import evaluate_one
from metalbench.static_check import StaticCheckResult
from metalbench.timing import TimingResult


def static_result(path: Path, *, ok: bool, errors: list[str] | None = None) -> StaticCheckResult:
    return StaticCheckResult(
        path=path,
        ok=ok,
        errors=errors or [],
        warnings=[],
        uses_compile_shader=ok,
        uses_load_metallib=False,
        found_metal_kernel_source=ok,
    )


def correctness_result(*, ok: bool, errors: list[str] | None = None) -> CorrectnessResult:
    return CorrectnessResult(
        ok=ok,
        trials=1,
        passed=1 if ok else 0,
        failed=0 if ok else 1,
        errors=errors or [],
    )


def timing_result(
    *,
    ok: bool,
    median_ms: float | None,
    errors: list[str] | None = None,
) -> TimingResult:
    return TimingResult(
        ok=ok,
        device="mps",
        median_ms=median_ms,
        mean_ms=median_ms,
        min_ms=median_ms,
        max_ms=median_ms,
        p25_ms=median_ms,
        p75_ms=median_ms,
        warmup=1,
        trials=1 if ok else 0,
        errors=errors or [],
    )


def test_static_check_failure_gates_runtime_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        eval_one_module.static_check,
        "check_generated_metal_kernel",
        lambda path: static_result(path, ok=False, errors=["missing metal loader"]),
    )

    def fail_stage(*args: object, **kwargs: object) -> object:
        raise AssertionError("runtime stage should not run")

    monkeypatch.setattr(eval_one_module.correctness, "run_correctness_trials", fail_stage)
    monkeypatch.setattr(eval_one_module.timing, "time_generated_metal_kernel", fail_stage)
    monkeypatch.setattr(eval_one_module.timing, "time_reference_mps_baseline", fail_stage)

    result = evaluate_one(ref_path, kernel_path, require_mps=False)

    assert result.static_check.ok is False
    assert result.correctness is None
    assert result.generated_timing is None
    assert result.mps_baseline_timing is None
    assert result.speedup_vs_mps is None
    assert result.metal_fast_0 is False
    assert result.metal_fast_1 is False
    assert result.metal_fast_2 is False
    assert "static_check: missing metal loader" in result.errors


def test_correctness_failure_gates_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        eval_one_module.static_check,
        "check_generated_metal_kernel",
        lambda path: static_result(path, ok=True),
    )
    monkeypatch.setattr(
        eval_one_module.correctness,
        "run_correctness_trials",
        lambda *args, **kwargs: correctness_result(ok=False, errors=["wrong output"]),
    )

    def fail_timing(*args: object, **kwargs: object) -> object:
        raise AssertionError("timing should not run")

    monkeypatch.setattr(eval_one_module.timing, "time_generated_metal_kernel", fail_timing)
    monkeypatch.setattr(eval_one_module.timing, "time_reference_mps_baseline", fail_timing)

    result = evaluate_one(ref_path, kernel_path, require_mps=False)

    assert result.static_check.ok is True
    assert result.correctness == correctness_result(ok=False, errors=["wrong output"])
    assert result.generated_timing is None
    assert result.mps_baseline_timing is None
    assert result.speedup_vs_mps is None
    assert result.metal_fast_0 is False
    assert "correctness: wrong output" in result.errors


def test_successful_evaluation_computes_speedup_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        eval_one_module.static_check,
        "check_generated_metal_kernel",
        lambda path: static_result(path, ok=True),
    )
    monkeypatch.setattr(
        eval_one_module.correctness,
        "run_correctness_trials",
        lambda *args, **kwargs: correctness_result(ok=True),
    )
    monkeypatch.setattr(
        eval_one_module.timing,
        "time_generated_metal_kernel",
        lambda *args, **kwargs: timing_result(ok=True, median_ms=2.0),
    )
    monkeypatch.setattr(
        eval_one_module.timing,
        "time_reference_mps_baseline",
        lambda *args, **kwargs: timing_result(ok=True, median_ms=5.0),
    )

    result = evaluate_one(
        ref_path,
        kernel_path,
        run_name="run",
        level=1,
        problem_id=19,
        sample_id=0,
        correctness_trials=3,
        perf_trials=4,
        warmup=2,
        require_mps=False,
    )

    assert result.run_name == "run"
    assert result.level == 1
    assert result.problem_id == 19
    assert result.sample_id == 0
    assert result.speedup_vs_mps == 2.5
    assert result.metal_fast_0 is True
    assert result.metal_fast_1 is True
    assert result.metal_fast_2 is True
    assert result.errors == []


@pytest.mark.parametrize(
    ("generated", "baseline", "expected_error"),
    [
        (
            timing_result(ok=False, median_ms=None, errors=["generated failed"]),
            timing_result(ok=True, median_ms=5.0),
            "generated_timing: generated failed",
        ),
        (
            timing_result(ok=True, median_ms=2.0),
            timing_result(ok=False, median_ms=None, errors=["baseline failed"]),
            "mps_baseline_timing: baseline failed",
        ),
    ],
)
def test_timing_failure_leaves_speedup_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generated: TimingResult,
    baseline: TimingResult,
    expected_error: str,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        eval_one_module.static_check,
        "check_generated_metal_kernel",
        lambda path: static_result(path, ok=True),
    )
    monkeypatch.setattr(
        eval_one_module.correctness,
        "run_correctness_trials",
        lambda *args, **kwargs: correctness_result(ok=True),
    )
    monkeypatch.setattr(
        eval_one_module.timing,
        "time_generated_metal_kernel",
        lambda *args, **kwargs: generated,
    )
    monkeypatch.setattr(
        eval_one_module.timing,
        "time_reference_mps_baseline",
        lambda *args, **kwargs: baseline,
    )

    result = evaluate_one(ref_path, kernel_path, require_mps=False)

    assert result.speedup_vs_mps is None
    assert result.metal_fast_0 is True
    assert result.metal_fast_1 is False
    assert result.metal_fast_2 is False
    assert expected_error in result.errors


def test_stage_exceptions_are_captured_in_result_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        eval_one_module.static_check,
        "check_generated_metal_kernel",
        lambda path: static_result(path, ok=True),
    )

    def raise_correctness(*args: object, **kwargs: object) -> CorrectnessResult:
        raise RuntimeError("import failed")

    monkeypatch.setattr(eval_one_module.correctness, "run_correctness_trials", raise_correctness)

    result = evaluate_one(ref_path, kernel_path, require_mps=False)

    assert result.correctness == CorrectnessResult(
        ok=False,
        trials=0,
        passed=0,
        failed=0,
        errors=["import failed"],
    )
    assert result.generated_timing is None
    assert result.mps_baseline_timing is None
    assert "correctness: import failed" in result.errors
