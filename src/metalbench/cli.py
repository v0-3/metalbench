import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from metalbench import env as env_module
from metalbench import kernelbench_adapter
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
