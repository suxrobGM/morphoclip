"""MorphoCLIP unified command-line interface.

Exposes a single ``morphoclip`` command (see ``[project.scripts]`` in
pyproject.toml) with grouped subcommands. Also runnable as a module
(``python -m morphoclip.cli ...``), which is how ``torchrun`` launches
distributed training.
"""

import typer

from morphoclip.cli import cellclip_cmds, data, features, text
from morphoclip.cli.benchmark import benchmark
from morphoclip.cli.evaluate import evaluate, infer, split
from morphoclip.cli.train import train

app = typer.Typer(
    name="morphoclip",
    no_args_is_help=True,
    add_completion=False,
    help="MorphoCLIP: text-supervised contrastive learning for perturbation matching.",
    pretty_exceptions_show_locals=False,
)

# Top-level commands.
app.command()(train)
app.command("eval")(evaluate)
app.command()(infer)
app.command()(split)
app.command()(benchmark)

# Grouped subcommands.
app.add_typer(data.app, name="data")
app.add_typer(features.app, name="features")
app.add_typer(text.app, name="text")
app.add_typer(cellclip_cmds.app, name="cellclip")

__all__ = ["app"]
