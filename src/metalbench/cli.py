import typer

app = typer.Typer(
    add_completion=False,
    help="Apple Metal-only kernel generation and evaluation harness.",
    name="metalbench",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    pass
