from pathlib import Path

import pytest

import metalbench.eval_batch as eval_batch_module
from metalbench.eval_batch import evaluate_run_directory
from metalbench.eval_one import EvalOneResult
from metalbench.static_check import StaticCheckResult


def failed_static_result(path: Path, errors: list[str]) -> StaticCheckResult:
    return StaticCheckResult(
        path=path,
        ok=False,
        errors=errors,
        warnings=[],
        uses_compile_shader=False,
        uses_load_metallib=False,
        found_metal_kernel_source=False,
    )


def batch_failure(
    *,
    run_name: str,
    level: int | None,
    problem_id: int | None,
    sample_id: int | None,
    ref_path: Path | None,
    kernel_path: Path,
    errors: list[str],
) -> EvalOneResult:
    return EvalOneResult(
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        sample_id=sample_id,
        ref_path=ref_path,
        kernel_path=kernel_path,
        static_check=failed_static_result(kernel_path, errors),
        correctness=None,
        generated_timing=None,
        mps_baseline_timing=None,
        speedup_vs_mps=None,
        metal_fast_0=False,
        metal_fast_1=False,
        metal_fast_2=False,
        errors=errors,
    )


def test_parse_kernel_filename_returns_metadata() -> None:
    metadata = eval_batch_module._parse_kernel_filename(
        Path("level_2_problem_17_sample_3_kernel.py")
    )

    assert metadata.level == 2
    assert metadata.problem_id == 17
    assert metadata.sample_id == 3


def test_invalid_filename_produces_failed_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    kernel_path = run_dir / "bad_kernel.py"
    kernel_path.write_text("", encoding="utf-8")

    results = evaluate_run_directory(run_dir, require_mps=False)

    assert len(results) == 1
    result = results[0]
    assert result.run_name == "run-a"
    assert result.level is None
    assert result.problem_id is None
    assert result.sample_id is None
    assert result.ref_path is None
    assert result.kernel_path == kernel_path
    assert result.static_check == failed_static_result(kernel_path, result.errors)
    assert result.metal_fast_0 is False
    assert result.metal_fast_1 is False
    assert result.metal_fast_2 is False
    assert result.errors
    assert "Malformed kernel filename" in result.errors[0]


def test_missing_kernelbench_reference_produces_failed_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    kernel_path = run_dir / "level_1_problem_19_sample_0_kernel.py"
    kernel_path.write_text("", encoding="utf-8")
    kernelbench_dir = tmp_path / "KernelBench"

    results = evaluate_run_directory(run_dir, kernelbench_dir=kernelbench_dir, require_mps=False)

    assert len(results) == 1
    result = results[0]
    assert result.run_name == "run-a"
    assert result.level == 1
    assert result.problem_id == 19
    assert result.sample_id == 0
    assert result.ref_path is None
    assert result.kernel_path == kernel_path
    assert result.static_check == failed_static_result(kernel_path, result.errors)
    assert result.errors == [f"KernelBench directory not found: {kernelbench_dir}"]
    assert result.metal_fast_0 is False
    assert result.metal_fast_1 is False
    assert result.metal_fast_2 is False


def test_multiple_kernels_are_evaluated_in_sorted_filename_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    first_kernel = run_dir / "level_1_problem_2_sample_0_kernel.py"
    second_kernel = run_dir / "level_1_problem_10_sample_0_kernel.py"
    ignored_file = run_dir / "level_1_problem_1_sample_0.py"
    for path in (second_kernel, first_kernel, ignored_file):
        path.write_text("", encoding="utf-8")
    first_ref = tmp_path / "2_ref.py"
    second_ref = tmp_path / "10_ref.py"
    seen_kernel_paths: list[Path] = []

    def fake_find_problem_file(level: int, problem_id: int, kernelbench_dir: Path) -> Path:
        return {2: first_ref, 10: second_ref}[problem_id]

    def fake_evaluate_one(
        ref_path: Path,
        kernel_path: Path,
        **kwargs: object,
    ) -> EvalOneResult:
        seen_kernel_paths.append(kernel_path)
        problem_id = 2 if kernel_path == first_kernel else 10
        return batch_failure(
            run_name="run-a",
            level=1,
            problem_id=problem_id,
            sample_id=0,
            ref_path=ref_path,
            kernel_path=kernel_path,
            errors=[],
        )

    monkeypatch.setattr(
        eval_batch_module.kernelbench_adapter,
        "find_problem_file",
        fake_find_problem_file,
    )
    monkeypatch.setattr(eval_batch_module.eval_one, "evaluate_one", fake_evaluate_one)

    results = evaluate_run_directory(run_dir, require_mps=False)

    assert seen_kernel_paths == [second_kernel, first_kernel]
    assert [result.kernel_path for result in results] == [second_kernel, first_kernel]
    assert [result.ref_path for result in results] == [second_ref, first_ref]


def test_evaluate_one_is_called_with_metadata_and_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    kernel_path = run_dir / "level_2_problem_17_sample_3_kernel.py"
    kernel_path.write_text("", encoding="utf-8")
    ref_path = tmp_path / "17_ref.py"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        eval_batch_module.kernelbench_adapter,
        "find_problem_file",
        lambda level, problem_id, kernelbench_dir: ref_path,
    )

    def fake_evaluate_one(
        ref_path: Path,
        kernel_path: Path,
        **kwargs: object,
    ) -> EvalOneResult:
        calls.append(
            {
                "ref_path": ref_path,
                "kernel_path": kernel_path,
                **kwargs,
            }
        )
        return batch_failure(
            run_name="run-a",
            level=2,
            problem_id=17,
            sample_id=3,
            ref_path=ref_path,
            kernel_path=kernel_path,
            errors=[],
        )

    monkeypatch.setattr(eval_batch_module.eval_one, "evaluate_one", fake_evaluate_one)

    evaluate_run_directory(
        run_dir,
        kernelbench_dir=tmp_path / "KernelBench",
        correctness_trials=7,
        perf_trials=11,
        warmup=13,
        require_mps=False,
    )

    assert calls == [
        {
            "ref_path": ref_path,
            "kernel_path": kernel_path,
            "run_name": "run-a",
            "level": 2,
            "problem_id": 17,
            "sample_id": 3,
            "correctness_trials": 7,
            "perf_trials": 11,
            "warmup": 13,
            "require_mps": False,
        }
    ]


def test_empty_run_directory_returns_empty_list(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()

    assert evaluate_run_directory(run_dir, require_mps=False) == []
