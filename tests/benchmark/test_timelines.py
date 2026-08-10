"""Tests for the one timeline helper.

`normalize_timelines` was written twice, in benchmark.stable_config and
cellclip.benchmark.export, and only the first copy was tested. The threshold
table was a third literal inside `get_timepoint_label`.
"""

import pytest

from benchmark.timelines import TIMELINE_CHOICES, get_timepoint_label, normalize_timelines


class TestNormalizeTimelines:
    def test_none_means_every_timeline(self) -> None:
        assert normalize_timelines(None) == list(TIMELINE_CHOICES)

    def test_a_bare_string_is_a_selection_of_one(self) -> None:
        assert normalize_timelines("short") == ["short"]

    def test_case_and_whitespace_are_ignored(self) -> None:
        assert normalize_timelines(["  LONG "]) == ["long"]

    def test_duplicates_collapse_and_order_survives(self) -> None:
        assert normalize_timelines(["long", "short", "long"]) == ["long", "short"]

    def test_an_empty_selection_means_every_timeline(self) -> None:
        assert normalize_timelines([]) == list(TIMELINE_CHOICES)

    def test_an_unknown_label_names_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="expected one of: short, long"):
            normalize_timelines(["medium"])


class TestTimepointLabel:
    @pytest.mark.parametrize(
        ("modality", "hours", "expected"),
        [
            ("compound", 24, "short"),
            ("compound", 48, "long"),
            ("orf", 48, "short"),
            ("orf", 96, "long"),
            ("crispr", 96, "short"),
            ("crispr", 144, "long"),
        ],
    )
    def test_the_threshold_moves_with_the_modality(
        self, modality: str, hours: int, expected: str
    ) -> None:
        assert get_timepoint_label(modality, hours) == expected

    def test_the_same_timepoint_reads_differently_per_modality(self) -> None:
        """48 hours is a long compound exposure and a short ORF one."""
        assert get_timepoint_label("compound", 48) == "long"
        assert get_timepoint_label("orf", 48) == "short"

    def test_an_unknown_modality_falls_back_to_the_slowest_threshold(self) -> None:
        assert get_timepoint_label("mystery", 96) == "short"
        assert get_timepoint_label("mystery", 97) == "long"
