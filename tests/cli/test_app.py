"""Structural tests for the Typer app.

Walking the tree and asking every command for its help is the cheapest net there
is: it catches a moved module, a broken Annotated option type, a renamed sub-app,
and an optional-extra import that leaked out of a command body.
"""

from collections.abc import Iterator

import pytest
import typer
from typer.testing import CliRunner

from morphoclip.cli import app

runner = CliRunner()


def command_paths(group: typer.Typer, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Every invocable command path under *group*, as argv fragments."""
    for command in group.registered_commands:
        name = command.name or (
            command.callback.__name__.replace("_", "-") if command.callback else ""
        )
        yield (*prefix, name)
    for sub in group.registered_groups:
        if sub.typer_instance is not None:
            yield from command_paths(sub.typer_instance, (*prefix, sub.name or ""))


ALL_COMMANDS = sorted(command_paths(app))
HELP_PATHS = [(), *ALL_COMMANDS]


def test_command_tree_is_stable() -> None:
    """The parametrize below reads the tree back off the app, so only this pins it."""
    assert [" ".join(path) for path in ALL_COMMANDS] == [
        "benchmark",
        "cellclip export",
        "cellclip pipeline",
        "cellclip train",
        "data check-plates",
        "data fetch",
        "eval",
        "export-profiles",
        "features download",
        "features extract",
        "features pipeline",
        "features repack",
        "features upload",
        "infer",
        "split",
        "text precompute",
        "train",
    ]


@pytest.mark.parametrize("path", HELP_PATHS, ids=[" ".join(p) or "root" for p in HELP_PATHS])
def test_help_exits_zero(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
