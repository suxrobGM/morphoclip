"""Structural tests for the Typer app.

`cli/` is the layer a file move breaks first, and the whole tree was untested.
Walking it and asking every command for its help is the cheapest net there is:
it catches a moved module, a broken Annotated option type, a renamed sub-app,
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


def test_command_tree_is_stable() -> None:
    """The full command surface, so a rename or a dropped registration is visible."""
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


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("path", ALL_COMMANDS, ids=" ".join)
def test_command_help_exits_zero(path: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
