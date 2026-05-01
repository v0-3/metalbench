from pathlib import Path

from pydantic import BaseModel

from metalbench import correctness, static_check, timing
from metalbench.correctness import CorrectnessResult
from metalbench.static_check import StaticCheckResult
from metalbench.timing import TimingResult


class EvalOneResult(BaseModel):
    run_name: str | None
    level: int | None
    problem_id: int | None
    sample_id: int | None
    ref_path: Path | None
    kernel_path: Path
    static_check: StaticCheckResult
    correctness: CorrectnessResult | None
    generated_timing: TimingResult | None
    mps_baseline_timing: TimingResult | None
    speedup_vs_mps: float | None
    metal_fast_0: bool
    metal_fast_1: bool
    metal_fast_2: bool
    errors: list[str]


def evaluate_one(
    ref_path: Path,
    kernel_path: Path,
    *,
    run_name: str | None = None,
    level: int | None = None,
    problem_id: int | None = None,
    sample_id: int | None = None,
    correctness_trials: int = 5,
    perf_trials: int = 100,
    warmup: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    require_mps: bool = True,
) -> EvalOneResult:
    errors: list[str] = []

    try:
        static_result = static_check.check_generated_metal_kernel(kernel_path)
    except Exception as exc:  # noqa: BLE001
        static_result = _static_exception_result(kernel_path, exc)

    errors.extend(_stage_errors("static_check", static_result.errors))
    if not static_result.ok:
        return _build_result(
            ref_path=ref_path,
            kernel_path=kernel_path,
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            sample_id=sample_id,
            static_result=static_result,
            correctness_result=None,
            generated_timing=None,
            baseline_timing=None,
            speedup_vs_mps=None,
            errors=errors,
        )

    try:
        correctness_result = correctness.run_correctness_trials(
            ref_path,
            kernel_path,
            trials=correctness_trials,
            rtol=rtol,
            atol=atol,
            require_mps=require_mps,
        )
    except Exception as exc:  # noqa: BLE001
        correctness_result = _correctness_exception_result(exc)

    errors.extend(_stage_errors("correctness", correctness_result.errors))
    if not correctness_result.ok:
        return _build_result(
            ref_path=ref_path,
            kernel_path=kernel_path,
            run_name=run_name,
            level=level,
            problem_id=problem_id,
            sample_id=sample_id,
            static_result=static_result,
            correctness_result=correctness_result,
            generated_timing=None,
            baseline_timing=None,
            speedup_vs_mps=None,
            errors=errors,
        )

    try:
        generated_timing = timing.time_generated_metal_kernel(
            ref_path,
            kernel_path,
            warmup=warmup,
            trials=perf_trials,
            require_mps=require_mps,
        )
    except Exception as exc:  # noqa: BLE001
        generated_timing = _timing_exception_result(warmup, exc)

    try:
        baseline_timing = timing.time_reference_mps_baseline(
            ref_path,
            warmup=warmup,
            trials=perf_trials,
            require_mps=require_mps,
        )
    except Exception as exc:  # noqa: BLE001
        baseline_timing = _timing_exception_result(warmup, exc)

    errors.extend(_stage_errors("generated_timing", generated_timing.errors))
    errors.extend(_stage_errors("mps_baseline_timing", baseline_timing.errors))
    speedup_vs_mps = _compute_speedup(generated_timing, baseline_timing)

    return _build_result(
        ref_path=ref_path,
        kernel_path=kernel_path,
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        sample_id=sample_id,
        static_result=static_result,
        correctness_result=correctness_result,
        generated_timing=generated_timing,
        baseline_timing=baseline_timing,
        speedup_vs_mps=speedup_vs_mps,
        errors=errors,
    )


def _build_result(
    *,
    ref_path: Path,
    kernel_path: Path,
    run_name: str | None,
    level: int | None,
    problem_id: int | None,
    sample_id: int | None,
    static_result: StaticCheckResult,
    correctness_result: CorrectnessResult | None,
    generated_timing: TimingResult | None,
    baseline_timing: TimingResult | None,
    speedup_vs_mps: float | None,
    errors: list[str],
) -> EvalOneResult:
    metal_fast_0 = static_result.ok and correctness_result is not None and correctness_result.ok
    metal_fast_1 = metal_fast_0 and speedup_vs_mps is not None and speedup_vs_mps > 1.0
    metal_fast_2 = metal_fast_0 and speedup_vs_mps is not None and speedup_vs_mps > 2.0

    return EvalOneResult(
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        sample_id=sample_id,
        ref_path=ref_path,
        kernel_path=kernel_path,
        static_check=static_result,
        correctness=correctness_result,
        generated_timing=generated_timing,
        mps_baseline_timing=baseline_timing,
        speedup_vs_mps=speedup_vs_mps,
        metal_fast_0=metal_fast_0,
        metal_fast_1=metal_fast_1,
        metal_fast_2=metal_fast_2,
        errors=errors,
    )


def _compute_speedup(
    generated_timing: TimingResult,
    baseline_timing: TimingResult,
) -> float | None:
    generated_median_ms = generated_timing.median_ms
    baseline_median_ms = baseline_timing.median_ms
    if (
        not generated_timing.ok
        or not baseline_timing.ok
        or generated_median_ms is None
        or baseline_median_ms is None
        or generated_median_ms <= 0.0
    ):
        return None
    return baseline_median_ms / generated_median_ms


def _static_exception_result(path: Path, exc: Exception) -> StaticCheckResult:
    return StaticCheckResult(
        path=path,
        ok=False,
        errors=[str(exc)],
        warnings=[],
        uses_compile_shader=False,
        uses_load_metallib=False,
        found_metal_kernel_source=False,
    )


def _correctness_exception_result(exc: Exception) -> CorrectnessResult:
    return CorrectnessResult(
        ok=False,
        trials=0,
        passed=0,
        failed=0,
        errors=[str(exc)],
    )


def _timing_exception_result(warmup: int, exc: Exception) -> TimingResult:
    return TimingResult(
        ok=False,
        device="mps",
        median_ms=None,
        mean_ms=None,
        min_ms=None,
        max_ms=None,
        p25_ms=None,
        p75_ms=None,
        warmup=warmup,
        trials=0,
        errors=[str(exc)],
    )


def _stage_errors(stage: str, errors: list[str]) -> list[str]:
    return [f"{stage}: {error}" for error in errors]


__all__ = [
    "EvalOneResult",
    "evaluate_one",
]
