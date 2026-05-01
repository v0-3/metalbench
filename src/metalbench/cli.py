import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer

import metalbench.env as env_module
from metalbench import eval_batch, eval_one, kernelbench_adapter, static_check
from metalbench.types import KBProblem

T = TypeVar("T")

app = typer.Typer(
    add_completion=False,
    help="Apple Metal-only kernel generation and evaluation harness.",
    name="metalbench",
    no_args_is_help=True,
)
kb_app = typer.Typer(
    add_completion=False,
    help="Inspect project-root KernelBench reference problems.",
    no_args_is_help=True,
)
app.add_typer(kb_app, name="kb")


@app.callback()
def main() -> None:
    pass


@app.command("env")
def show_environment(
    require_mps: Annotated[
        bool,
        typer.Option(
            "--require-mps",
            help="Exit with an error when Apple Metal MPS is unavailable.",
        ),
    ] = False,
) -> None:
    if require_mps:
        try:
            env_module.require_mps()
        except RuntimeError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error

    typer.echo(json.dumps(env_module.describe_environment(), indent=2))


@app.command("check")
def check_generated_kernel(path: Path) -> None:
    try:
        result = static_check.check_generated_metal_kernel(path)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error

    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    if not result.ok:
        raise typer.Exit(1)


@app.command("eval-one")
def evaluate_one_kernel(
    ref_path: Annotated[
        Path,
        typer.Option(
            "--ref-path",
            help="Reference KernelBench problem path.",
        ),
    ],
    kernel_path: Annotated[
        Path,
        typer.Option(
            "--kernel-path",
            help="Generated Metal kernel path.",
        ),
    ],
    run_name: Annotated[
        str | None,
        typer.Option(
            "--run-name",
            help="Optional run name to include in JSON output.",
        ),
    ] = None,
    level: Annotated[
        int | None,
        typer.Option(
            "--level",
            help="Optional KernelBench level to include in JSON output.",
        ),
    ] = None,
    problem_id: Annotated[
        int | None,
        typer.Option(
            "--problem-id",
            help="Optional KernelBench problem id to include in JSON output.",
        ),
    ] = None,
    sample_id: Annotated[
        int | None,
        typer.Option(
            "--sample-id",
            help="Optional sample id to include in JSON output.",
        ),
    ] = None,
    correctness_trials: Annotated[
        int,
        typer.Option(
            "--correctness-trials",
            help="Correctness trials to run after static checks pass.",
        ),
    ] = 5,
    perf_trials: Annotated[
        int,
        typer.Option(
            "--perf-trials",
            help="Timing trials to run after correctness passes.",
        ),
    ] = 100,
    warmup: Annotated[
        int,
        typer.Option(
            "--warmup",
            help="Timing warmup iterations.",
        ),
    ] = 10,
    rtol: Annotated[
        float,
        typer.Option(
            "--rtol",
            help="Relative tolerance for correctness checks.",
        ),
    ] = 1e-4,
    atol: Annotated[
        float,
        typer.Option(
            "--atol",
            help="Absolute tolerance for correctness checks.",
        ),
    ] = 1e-4,
    require_mps: Annotated[
        bool,
        typer.Option(
            "--require-mps",
            help="Exit with an error when Apple Metal MPS is unavailable.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Optional JSON output path.",
        ),
    ] = None,
) -> None:
    if require_mps:
        try:
            env_module.require_mps()
        except RuntimeError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error

    result = eval_one.evaluate_one(
        ref_path,
        kernel_path,
        run_name=run_name,
        level=level,
        problem_id=problem_id,
        sample_id=sample_id,
        correctness_trials=correctness_trials,
        perf_trials=perf_trials,
        warmup=warmup,
        rtol=rtol,
        atol=atol,
        require_mps=require_mps,
    )
    result_json = json.dumps(result.model_dump(mode="json"), indent=2)
    if output is None:
        typer.echo(result_json)
        return

    output.write_text(f"{result_json}\n", encoding="utf-8")


@app.command("eval-run")
def evaluate_run_directory(
    run_dir: Annotated[
        Path,
        typer.Option(
            "--run-dir",
            help="Run directory containing generated kernel files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="Required JSON output path.",
        ),
    ],
    kernelbench_dir: Annotated[
        Path,
        typer.Option(
            "--kernelbench-dir",
            help="Project-root KernelBench problem directory.",
        ),
    ] = Path("KernelBench"),
    correctness_trials: Annotated[
        int,
        typer.Option(
            "--correctness-trials",
            help="Correctness trials to run after static checks pass.",
        ),
    ] = 5,
    perf_trials: Annotated[
        int,
        typer.Option(
            "--perf-trials",
            help="Timing trials to run after correctness passes.",
        ),
    ] = 100,
    warmup: Annotated[
        int,
        typer.Option(
            "--warmup",
            help="Timing warmup iterations.",
        ),
    ] = 10,
    require_mps: Annotated[
        bool,
        typer.Option(
            "--require-mps",
            help="Exit with an error when Apple Metal MPS is unavailable.",
        ),
    ] = False,
) -> None:
    if require_mps:
        try:
            env_module.require_mps()
        except RuntimeError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error

    results = eval_batch.evaluate_run_directory(
        run_dir,
        kernelbench_dir,
        correctness_trials=correctness_trials,
        perf_trials=perf_trials,
        warmup=warmup,
        require_mps=require_mps,
    )
    result_json = json.dumps([result.model_dump(mode="json") for result in results], indent=2)
    output.write_text(f"{result_json}\n", encoding="utf-8")


@kb_app.command("list")
def list_kernelbench_problems(
    kernelbench_dir: Annotated[
        Path,
        typer.Option(
            "--kernelbench-dir",
            help="Project-root KernelBench problem directory.",
        ),
    ] = Path("KernelBench"),
    level: Annotated[
        int,
        typer.Option(
            "--level",
            help="KernelBench problem level.",
        ),
    ] = 1,
) -> None:
    problems = _run_kb_action(lambda: kernelbench_adapter.list_problems(level, kernelbench_dir))
    typer.echo(
        json.dumps(
            [_problem_json(problem, print_source=False) for problem in problems],
            indent=2,
        )
    )


@kb_app.command("show")
def show_kernelbench_problem(
    kernelbench_dir: Annotated[
        Path,
        typer.Option(
            "--kernelbench-dir",
            help="Project-root KernelBench problem directory.",
        ),
    ] = Path("KernelBench"),
    level: Annotated[
        int,
        typer.Option(
            "--level",
            help="KernelBench problem level.",
        ),
    ] = 1,
    problem_id: Annotated[
        int,
        typer.Option(
            "--problem-id",
            help="KernelBench problem id.",
        ),
    ] = 19,
    print_source: Annotated[
        bool,
        typer.Option(
            "--print-source",
            help="Include reference problem source in JSON output.",
        ),
    ] = False,
) -> None:
    problem = _run_kb_action(
        lambda: kernelbench_adapter.load_problem(level, problem_id, kernelbench_dir)
    )
    typer.echo(json.dumps(_problem_json(problem, print_source=print_source), indent=2))


def _run_kb_action(action: Callable[[], T]) -> T:
    try:
        return action()
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


def _problem_json(problem: KBProblem, print_source: bool) -> dict[str, object]:
    if print_source:
        return problem.model_dump(mode="json")
    return problem.model_dump(mode="json", exclude={"source"})
