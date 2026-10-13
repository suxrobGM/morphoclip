"""Typer choice for the ``text_level`` argument of ``generate_text``."""

from enum import StrEnum


class TextLevel(StrEnum):
    name_only = "name_only"
    name_target = "name_target"
    full = "full"
