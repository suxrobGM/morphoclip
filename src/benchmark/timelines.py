"""Timeline labels: the modality-aware short/long split CPJUMP1 slices by.

The thresholds differ per modality because a compound acts within a day while a
CRISPR knockout needs several, so "48 hours" is a long compound timepoint and a
short CRISPR one.
"""

from typing import Any

TIMELINE_CHOICES = ("short", "long")

# Hours at or below which a timepoint counts as short, per perturbation type.
SHORT_TIMEPOINT_HOURS = {"compound": 24, "orf": 48, "crispr": 96}
DEFAULT_SHORT_HOURS = 96


def get_timepoint_label(modality: str, hours: int) -> str:
    """Return ``"short"`` or ``"long"`` for a timepoint in *hours*."""
    return "short" if hours <= SHORT_TIMEPOINT_HOURS.get(modality, DEFAULT_SHORT_HOURS) else "long"


def normalize_timelines(value: Any) -> list[str]:
    """Normalize a timeline selection to a deduplicated, validated list.

    Args:
        value: None for every timeline, a single label, or a sequence of labels.
            Case and surrounding whitespace are ignored.

    Returns:
        The requested labels in the order given, without duplicates.

    Raises:
        ValueError: If a label is not one of :data:`TIMELINE_CHOICES`.
    """
    if value is None:
        return list(TIMELINE_CHOICES)

    values = [value] if isinstance(value, str) else list(value)

    normalized: list[str] = []
    for item in values:
        label = str(item).strip().lower()
        if label not in TIMELINE_CHOICES:
            choices = ", ".join(TIMELINE_CHOICES)
            raise ValueError(f"Invalid timeline {item!r}; expected one of: {choices}")
        if label not in normalized:
            normalized.append(label)

    return normalized or list(TIMELINE_CHOICES)
