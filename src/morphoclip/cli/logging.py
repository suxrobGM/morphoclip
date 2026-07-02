"""Shared logging setup for CLI commands."""

import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    """Route library-level ``logging`` calls through a Rich handler.

    Idempotent and safe to call at the start of any command. Uses ``force=True``
    so repeated invocations reconfigure a single handler rather than stacking.

    Args:
        level: Root logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False)],
        force=True,
    )
