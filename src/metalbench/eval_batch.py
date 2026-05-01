import re
from dataclasses import dataclass
from pathlib import Path

from metalbench import eval_one, kernelbench_adapter
from metalbench.eval_one import EvalOneResult
from metalbench.static_check import StaticCheckResult

_KERNEL_FILENAME_RE = re.compile(
    r"level_(?P<level>\d+)_problem_(?P<problem_id>\d+)_sample_(?P<sample_id>\d+)_kernel\.py"
)


@dataclass(frozen=True)
class KernelMetadata:
    level: int
    problem_id: int
    sample_id: int


def evaluate_run_directory(
    run_dir: Path,
    kernelbench_dir: Path = Path("KernelBench"),
    *,
    correctness_trials: int = 5,
    perf_trials: int = 100,
    warmup: int = 10,
    require_mps: bool = True,
) -> list[EvalOneResult]:
    results: list[EvalOneResult] = []

    for kernel_path in sorted(run_dir.glob("*_kernel.py")):
        try:
            metadata = _parse_kernel_filename(kernel_path)
        except ValueError as error:
            results.append(_parser_failure_result(run_dir, kernel_path, error))
            continue

        try:
            ref_path = kernelbench_adapter.find_problem_file(
                metadata.level,
                metadata.problem_id,
                kernelbench_dir,
            )
        except (FileNotFoundError, ValueError) as error:
            results.append(_missing_reference_result(run_dir, kernel_path, metadata, error))
            continue

        results.append(
            eval_one.evaluate_one(
                ref_path,
                kernel_path,
                run_name=run_dir.name,
                level=metadata.level,
                problem_id=metadata.problem_id,
                sample_id=metadata.sample_id,
                correctness_trials=correctness_trials,
                perf_trials=perf_trials,
                warmup=warmup,
                require_mps=require_mps,
            )
        )

    return results


def _parse_kernel_filename(path: Path) -> KernelMetadata:
    match = _KERNEL_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"Malformed kernel filename: {path.name}; expected "
            "level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py"
        )

    return KernelMetadata(
        level=int(match.group("level")),
        problem_id=int(match.group("problem_id")),
        sample_id=int(match.group("sample_id")),
    )


def _parser_failure_result(
    run_dir: Path,
    kernel_path: Path,
    error: ValueError,
) -> EvalOneResult:
    return _failure_result(
        run_name=run_dir.name,
        level=None,
        problem_id=None,
        sample_id=None,
        ref_path=None,
        kernel_path=kernel_path,
        errors=[str(error)],
    )


def _missing_reference_result(
    run_dir: Path,
    kernel_path: Path,
    metadata: KernelMetadata,
    error: FileNotFoundError | ValueError,
) -> EvalOneResult:
    return _failure_result(
        run_name=run_dir.name,
        level=metadata.level,
        problem_id=metadata.problem_id,
        sample_id=metadata.sample_id,
        ref_path=None,
        kernel_path=kernel_path,
        errors=[str(error)],
    )


def _failure_result(
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
        static_check=StaticCheckResult(
            path=kernel_path,
            ok=False,
            errors=errors,
            warnings=[],
            uses_compile_shader=False,
            uses_load_metallib=False,
            found_metal_kernel_source=False,
        ),
        correctness=None,
        generated_timing=None,
        mps_baseline_timing=None,
        speedup_vs_mps=None,
        metal_fast_0=False,
        metal_fast_1=False,
        metal_fast_2=False,
        errors=errors,
    )


__all__ = ["evaluate_run_directory"]
