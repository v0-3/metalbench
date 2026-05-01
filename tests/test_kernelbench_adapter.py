from hashlib import sha256
from pathlib import Path

import pytest

from metalbench.kernelbench_adapter import find_problem_file, list_problems, load_problem
from metalbench.types import KBProblem


def write_problem(
    kernelbench_dir: Path,
    level: int,
    problem_id: int,
    name: str,
    source: str,
) -> Path:
    level_dir = kernelbench_dir / f"level{level}"
    level_dir.mkdir(parents=True, exist_ok=True)
    path = level_dir / f"{problem_id}_{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_find_problem_file_returns_matching_path(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    expected = write_problem(kernelbench_dir, 1, 19, "ReLU", "class Model:\n    pass\n")

    assert find_problem_file(1, 19, kernelbench_dir) == expected


def test_find_problem_file_raises_when_kernelbench_dir_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="KernelBench directory not found"):
        find_problem_file(1, 19, tmp_path / "missing")


def test_find_problem_file_raises_when_level_dir_is_missing(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    kernelbench_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="KernelBench level directory not found"):
        find_problem_file(2, 19, kernelbench_dir)


def test_find_problem_file_raises_when_problem_is_missing(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    (kernelbench_dir / "level1").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="KernelBench problem not found"):
        find_problem_file(1, 19, kernelbench_dir)


def test_find_problem_file_raises_when_multiple_files_match(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    write_problem(kernelbench_dir, 1, 19, "ReLU", "one")
    write_problem(kernelbench_dir, 1, 19, "ReLU_copy", "two")

    with pytest.raises(ValueError, match="Multiple KernelBench problems match"):
        find_problem_file(1, 19, kernelbench_dir)


def test_load_problem_returns_typed_metadata_and_source_hash(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "class Model:\n    pass\n"
    path = write_problem(kernelbench_dir, 1, 19, "ReLU", source)

    problem = load_problem(1, 19, kernelbench_dir)

    assert isinstance(problem, KBProblem)
    assert problem.level == 1
    assert problem.problem_id == 19
    assert problem.name == "ReLU"
    assert problem.path == path
    assert problem.source == source
    assert problem.source_sha256 == sha256(source.encode("utf-8")).hexdigest()


def test_list_problems_returns_problem_metadata_sorted_by_problem_id(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    write_problem(kernelbench_dir, 1, 10, "Matmul", "ten")
    write_problem(kernelbench_dir, 1, 2, "ReLU", "two")
    write_problem(kernelbench_dir, 2, 1, "Other", "other")

    problems = list_problems(1, kernelbench_dir)

    assert [problem.problem_id for problem in problems] == [2, 10]
    assert [problem.name for problem in problems] == ["ReLU", "Matmul"]
