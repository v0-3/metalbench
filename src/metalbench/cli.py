import json
from typing import Annotated

import typer

from metalbench import env as env_module

app = typer.Typer(
    add_completion=False,
    help="Apple Metal-only kernel generation and evaluation harness.",
    name="metalbench",
    no_args_is_help=True,
)


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
