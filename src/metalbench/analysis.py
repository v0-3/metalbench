import json
import math
from pathlib import Path
from statistics import median
from typing import TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from metalbench.eval_one import EvalOneResult

SpeedupRow: TypeAlias = dict[str, str | int | float | None]

FAILURE_BUCKETS = (
    "malformed_filename",
    "missing_reference",
    "static_check_failed",
    "correctness_failed",
    "generated_timing_failed",
    "mps_baseline_failed",
)


class RunAnalysis(BaseModel):
    run_dir: Path
    total: int
    static_ok: int
    correct: int
    timed_generated: int
    timed_baseline: int
    metal_fast_0: float
    metal_fast_1: float
    metal_fast_2: float
    median_speedup_correct: float | None
    geomean_speedup_correct: float | None
    best_speedups: list[SpeedupRow]
    worst_speedups: list[SpeedupRow]
    failures_by_type: dict[str, int]


def analyze_eval_results(eval_results_path: Path) -> RunAnalysis:
    raw_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    if not isinstance(raw_results, list):
        raise ValueError("Eval results JSON must be a JSON list")

    try:
        results = TypeAdapter(list[EvalOneResult]).validate_python(raw_results)
    except ValidationError as error:
        raise ValueError(f"Invalid eval result rows: {error}") from error

    total = len(results)
    correct_results = [result for result in results if _is_correct(result)]
    valid_speedup_results = [
        result for result in correct_results if _has_valid_speedup_timing(result)
    ]
    speedups = [
        result.speedup_vs_mps
        for result in valid_speedup_results
        if result.speedup_vs_mps is not None
    ]

    return RunAnalysis(
        run_dir=eval_results_path.parent,
        total=total,
        static_ok=sum(1 for result in results if result.static_check.ok),
        correct=len(correct_results),
        timed_generated=sum(
            1
            for result in results
            if result.generated_timing is not None and result.generated_timing.ok
        ),
        timed_baseline=sum(
            1
            for result in results
            if result.mps_baseline_timing is not None and result.mps_baseline_timing.ok
        ),
        metal_fast_0=_fraction(len(correct_results), total),
        metal_fast_1=_fraction(
            sum(
                1
                for result in correct_results
                if result.speedup_vs_mps is not None and result.speedup_vs_mps > 1.0
            ),
            total,
        ),
        metal_fast_2=_fraction(
            sum(
                1
                for result in correct_results
                if result.speedup_vs_mps is not None and result.speedup_vs_mps > 2.0
            ),
            total,
        ),
        median_speedup_correct=median(speedups) if speedups else None,
        geomean_speedup_correct=_geomean([speedup for speedup in speedups if speedup > 0.0]),
        best_speedups=[
            _speedup_row(result)
            for result in sorted(
                valid_speedup_results,
                key=lambda result: result.speedup_vs_mps or 0.0,
                reverse=True,
            )[:5]
        ],
        worst_speedups=[
            _speedup_row(result)
            for result in sorted(
                valid_speedup_results,
                key=lambda result: result.speedup_vs_mps or 0.0,
            )[:5]
        ],
        failures_by_type=_failure_counts(results),
    )


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_correct(result: EvalOneResult) -> bool:
    return (
        result.static_check.ok
        and result.correctness is not None
        and result.correctness.ok
    )


def _has_valid_speedup_timing(result: EvalOneResult) -> bool:
    return (
        result.generated_timing is not None
        and result.generated_timing.ok
        and result.mps_baseline_timing is not None
        and result.mps_baseline_timing.ok
        and result.speedup_vs_mps is not None
    )


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _speedup_row(result: EvalOneResult) -> SpeedupRow:
    return {
        "run_name": result.run_name,
        "level": result.level,
        "problem_id": result.problem_id,
        "sample_id": result.sample_id,
        "kernel_path": str(result.kernel_path),
        "speedup_vs_mps": result.speedup_vs_mps,
    }


def _failure_counts(results: list[EvalOneResult]) -> dict[str, int]:
    counts = dict.fromkeys(FAILURE_BUCKETS, 0)
    for result in results:
        is_malformed = _is_malformed_filename(result)
        is_missing_reference = _is_missing_reference(result)

        if is_malformed:
            counts["malformed_filename"] += 1
        if is_missing_reference:
            counts["missing_reference"] += 1
        if not result.static_check.ok and not is_malformed and not is_missing_reference:
            counts["static_check_failed"] += 1
        if result.static_check.ok and result.correctness is not None and not result.correctness.ok:
            counts["correctness_failed"] += 1
        if _is_correct(result) and (
            result.generated_timing is None or not result.generated_timing.ok
        ):
            counts["generated_timing_failed"] += 1
        if _is_correct(result) and (
            result.mps_baseline_timing is None or not result.mps_baseline_timing.ok
        ):
            counts["mps_baseline_failed"] += 1

    return counts


def _is_malformed_filename(result: EvalOneResult) -> bool:
    return _has_error_containing(result, "Malformed kernel filename")


def _is_missing_reference(result: EvalOneResult) -> bool:
    return result.ref_path is None and _has_error_containing(result, "KernelBench")


def _has_error_containing(result: EvalOneResult, text: str) -> bool:
    return any(text in error for error in result.errors + result.static_check.errors)


__all__ = [
    "RunAnalysis",
    "analyze_eval_results",
]
