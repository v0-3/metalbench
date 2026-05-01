import json
from pathlib import Path

import pytest

from metalbench.analysis import analyze_eval_results
from metalbench.correctness import CorrectnessResult
from metalbench.eval_one import EvalOneResult
from metalbench.static_check import StaticCheckResult
from metalbench.timing import TimingResult

DEFAULT_REF_PATH = object()


def write_results(path: Path, results: list[EvalOneResult]) -> None:
    path.write_text(
        json.dumps([result.model_dump(mode="json") for result in results]),
        encoding="utf-8",
    )


def timing_result(ok: bool, median_ms: float | None = 1.0) -> TimingResult:
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
        errors=[] if ok else ["timing failed"],
    )


def eval_result(
    tmp_path: Path,
    *,
    index: int,
    static_ok: bool = True,
    correctness_ok: bool | None = True,
    generated_timing: TimingResult | None = None,
    baseline_timing: TimingResult | None = None,
    speedup_vs_mps: float | None = None,
    errors: list[str] | None = None,
    ref_path: Path | None | object = DEFAULT_REF_PATH,
) -> EvalOneResult:
    kernel_path = tmp_path / f"level_1_problem_{index}_sample_0_kernel.py"
    static_check = StaticCheckResult(
        path=kernel_path,
        ok=static_ok,
        errors=[] if static_ok else (errors or ["static failed"]),
        warnings=[],
        uses_compile_shader=static_ok,
        uses_load_metallib=False,
        found_metal_kernel_source=static_ok,
    )
    correctness = (
        None
        if correctness_ok is None
        else CorrectnessResult(
            ok=correctness_ok,
            trials=1,
            passed=1 if correctness_ok else 0,
            failed=0 if correctness_ok else 1,
            errors=[] if correctness_ok else ["wrong output"],
        )
    )
    metal_fast_0 = static_ok and correctness_ok is True
    metal_fast_1 = metal_fast_0 and speedup_vs_mps is not None and speedup_vs_mps > 1.0
    metal_fast_2 = metal_fast_0 and speedup_vs_mps is not None and speedup_vs_mps > 2.0
    resolved_ref_path = tmp_path / f"{index}_ref.py" if ref_path is DEFAULT_REF_PATH else ref_path
    return EvalOneResult(
        run_name="run-a",
        level=1,
        problem_id=index,
        sample_id=0,
        ref_path=resolved_ref_path,
        kernel_path=kernel_path,
        static_check=static_check,
        correctness=correctness,
        generated_timing=generated_timing,
        mps_baseline_timing=baseline_timing,
        speedup_vs_mps=speedup_vs_mps,
        metal_fast_0=metal_fast_0,
        metal_fast_1=metal_fast_1,
        metal_fast_2=metal_fast_2,
        errors=errors or [],
    )


def test_analyze_empty_result_list(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    results_path.write_text("[]", encoding="utf-8")

    analysis = analyze_eval_results(results_path)

    assert analysis.run_dir == tmp_path
    assert analysis.total == 0
    assert analysis.static_ok == 0
    assert analysis.correct == 0
    assert analysis.timed_generated == 0
    assert analysis.timed_baseline == 0
    assert analysis.metal_fast_0 == 0.0
    assert analysis.metal_fast_1 == 0.0
    assert analysis.metal_fast_2 == 0.0
    assert analysis.median_speedup_correct is None
    assert analysis.geomean_speedup_correct is None
    assert analysis.best_speedups == []
    assert analysis.worst_speedups == []


def test_analyze_exact_metric_fractions(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    write_results(
        results_path,
        [
            eval_result(
                tmp_path,
                index=1,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(True),
                speedup_vs_mps=3.0,
            ),
            eval_result(
                tmp_path,
                index=2,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(True),
                speedup_vs_mps=1.5,
            ),
            eval_result(
                tmp_path,
                index=3,
                static_ok=True,
                correctness_ok=False,
            ),
            eval_result(
                tmp_path,
                index=4,
                static_ok=False,
                correctness_ok=None,
            ),
        ],
    )

    analysis = analyze_eval_results(results_path)

    assert analysis.total == 4
    assert analysis.static_ok == 3
    assert analysis.correct == 2
    assert analysis.timed_generated == 2
    assert analysis.timed_baseline == 2
    assert analysis.metal_fast_0 == 0.5
    assert analysis.metal_fast_1 == 0.5
    assert analysis.metal_fast_2 == 0.25


def test_analyze_median_and_geomean_speedups(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    write_results(
        results_path,
        [
            eval_result(
                tmp_path,
                index=1,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(True),
                speedup_vs_mps=2.0,
            ),
            eval_result(
                tmp_path,
                index=2,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(True),
                speedup_vs_mps=8.0,
            ),
            eval_result(
                tmp_path,
                index=3,
                generated_timing=timing_result(False),
                baseline_timing=timing_result(True),
                speedup_vs_mps=100.0,
            ),
        ],
    )

    analysis = analyze_eval_results(results_path)

    assert analysis.median_speedup_correct == 5.0
    assert analysis.geomean_speedup_correct == 4.0


def test_analyze_best_and_worst_speedup_ordering(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    speeds = [1.2, 4.0, 2.5, 0.75, 6.0, 3.0]
    write_results(
        results_path,
        [
            eval_result(
                tmp_path,
                index=index,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(True),
                speedup_vs_mps=speed,
            )
            for index, speed in enumerate(speeds, start=1)
        ],
    )

    analysis = analyze_eval_results(results_path)

    assert [row["speedup_vs_mps"] for row in analysis.best_speedups] == [6.0, 4.0, 3.0, 2.5, 1.2]
    assert [row["speedup_vs_mps"] for row in analysis.worst_speedups] == [0.75, 1.2, 2.5, 3.0, 4.0]
    assert analysis.best_speedups[0] == {
        "run_name": "run-a",
        "level": 1,
        "problem_id": 5,
        "sample_id": 0,
        "kernel_path": str(tmp_path / "level_1_problem_5_sample_0_kernel.py"),
        "speedup_vs_mps": 6.0,
    }


def test_analyze_failure_buckets(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    write_results(
        results_path,
        [
            eval_result(
                tmp_path,
                index=1,
                static_ok=False,
                correctness_ok=None,
                errors=["Malformed kernel filename: bad.py"],
                ref_path=None,
            ),
            eval_result(
                tmp_path,
                index=2,
                static_ok=False,
                correctness_ok=None,
                errors=["KernelBench problem not found: level=1 problem_id=2"],
                ref_path=None,
            ),
            eval_result(
                tmp_path,
                index=3,
                static_ok=False,
                correctness_ok=None,
                errors=["Missing required token: class ModelNew"],
            ),
            eval_result(
                tmp_path,
                index=4,
                correctness_ok=False,
            ),
            eval_result(
                tmp_path,
                index=5,
                generated_timing=None,
                baseline_timing=timing_result(True),
                speedup_vs_mps=None,
            ),
            eval_result(
                tmp_path,
                index=6,
                generated_timing=timing_result(True),
                baseline_timing=timing_result(False),
                speedup_vs_mps=None,
            ),
        ],
    )

    analysis = analyze_eval_results(results_path)

    assert analysis.failures_by_type == {
        "malformed_filename": 1,
        "missing_reference": 1,
        "static_check_failed": 1,
        "correctness_failed": 1,
        "generated_timing_failed": 1,
        "mps_baseline_failed": 1,
    }


def test_analyze_invalid_json_shape_raises_value_error(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    results_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON list"):
        analyze_eval_results(results_path)


def test_analyze_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_eval_results(tmp_path / "missing.json")
