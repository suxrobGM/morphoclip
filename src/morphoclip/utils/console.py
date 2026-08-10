"""The one Rich console, and the logging setup that shares it.

Only ``morphoclip.cli`` and dedicated reporting modules should write to stdout.
Library code logs instead, and :func:`setup_logging` routes those records
through the same console so output interleaves in order.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

FILE_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
FILE_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int = logging.INFO, log_path: Path | None = None) -> None:
    """Route library ``logging`` calls through Rich, optionally teeing to a file.

    Idempotent: repeated calls replace the handlers rather than stacking them,
    so a command can call it without knowing whether something else already did.

    Args:
        level: Level for console output.
        log_path: When given, also write DEBUG records here. Used by the
            unattended extraction pipeline, where the console scrollback is not
            enough to diagnose an overnight run.
    """
    handlers: list[logging.Handler] = [
        RichHandler(console=console, show_time=False, show_path=False, level=level)
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, FILE_LOG_DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.DEBUG if log_path is not None else level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


def make_progress(*, transient: bool = False) -> Progress:
    """The bar for long-running work: description, percent, elapsed and remaining.

    The training loop and the plate scan build their own, deliberately: one
    disables itself on non-main ranks, the other counts plates and finishes too
    fast for a time estimate to mean anything.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )
