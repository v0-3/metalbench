from hashlib import sha256
from pathlib import Path

from metalbench.types import KBProblem


def find_problem_file(
    level: int,
    problem_id: int,
    kernelbench_dir: Path = Path("KernelBench"),
) -> Path:
    if not kernelbench_dir.is_dir():
        raise FileNotFoundError(f"KernelBench directory not found: {kernelbench_dir}")

    level_dir = kernelbench_dir / f"level{level}"
    if not level_dir.is_dir():
        raise FileNotFoundError(f"KernelBench level directory not found: {level_dir}")

    matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
    if not matches:
        raise FileNotFoundError(
            f"KernelBench problem not found: level={level} problem_id={problem_id}"
        )
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple KernelBench problems match level={level} problem_id={problem_id}: {paths}"
        )

    return matches[0]


def load_problem(
    level: int,
    problem_id: int,
    kernelbench_dir: Path = Path("KernelBench"),
) -> KBProblem:
    path = find_problem_file(level, problem_id, kernelbench_dir)
    source_bytes = path.read_bytes()

    return KBProblem(
        level=level,
        problem_id=problem_id,
        name=_problem_name(path),
        path=path,
        source=source_bytes.decode("utf-8"),
        source_sha256=sha256(source_bytes).hexdigest(),
    )


def list_problems(
    level: int,
    kernelbench_dir: Path = Path("KernelBench"),
) -> list[KBProblem]:
    if not kernelbench_dir.is_dir():
        raise FileNotFoundError(f"KernelBench directory not found: {kernelbench_dir}")

    level_dir = kernelbench_dir / f"level{level}"
    if not level_dir.is_dir():
        raise FileNotFoundError(f"KernelBench level directory not found: {level_dir}")

    return [
        load_problem(level, _problem_id(path), kernelbench_dir)
        for path in sorted(level_dir.glob("*_*.py"), key=_problem_id)
    ]


def _problem_id(path: Path) -> int:
    return int(path.stem.split("_", 1)[0])


def _problem_name(path: Path) -> str:
    return path.stem.split("_", 1)[1]


__all__ = ["find_problem_file", "list_problems", "load_problem"]
