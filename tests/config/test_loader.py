"""Tests for the shared YAML loader in `morphoclip.config`.

The `extends` chain, dotted overrides and strict-model rejection used to be
written twice (once per training package) and tested through whichever config
happened to exercise them. Testing the loader directly is what lets the two
packages stop carrying their own copy.
"""

from pathlib import Path

import pytest
import yaml

from morphoclip.config import (
    StrictModel,
    apply_overrides,
    load_config,
    load_yaml_mapping,
    merge_dicts,
    parse_dotted_override,
    resolve_extends,
)


class Section(StrictModel):
    size: int = 1
    rate: float = 0.1
    label: str = "none"


class Root(StrictModel):
    section: Section = Section()


def write(path: Path, data: object) -> Path:
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


class TestMergeDicts:
    def test_nested_mappings_merge_key_by_key(self) -> None:
        merged = merge_dicts({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_a_scalar_replaces_a_mapping_wholesale(self) -> None:
        assert merge_dicts({"a": {"x": 1}}, {"a": 5}) == {"a": 5}

    def test_lists_replace_rather_than_concatenate(self) -> None:
        assert merge_dicts({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_inputs_are_left_alone(self) -> None:
        base = {"a": {"x": 1}}
        merge_dicts(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestLoadYamlMapping:
    def test_an_empty_file_is_an_empty_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_yaml_mapping(path) == {}

    def test_a_top_level_list_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "list.yaml", [1, 2, 3])
        with pytest.raises(ValueError, match="must be a mapping"):
            load_yaml_mapping(path)


class TestResolveExtends:
    def test_a_child_wins_over_its_parent(self, tmp_path: Path) -> None:
        write(tmp_path / "base.yaml", {"section": {"size": 1, "label": "base"}})
        child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "section": {"size": 2}})
        assert resolve_extends(child) == {"section": {"size": 2, "label": "base"}}

    def test_later_parents_win_over_earlier_ones(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", {"section": {"size": 1, "label": "a"}})
        write(tmp_path / "b.yaml", {"section": {"label": "b"}})
        child = write(tmp_path / "child.yaml", {"extends": ["a.yaml", "b.yaml"]})
        assert resolve_extends(child) == {"section": {"size": 1, "label": "b"}}

    def test_grandparents_resolve_through_the_chain(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", {"section": {"size": 1, "label": "a"}})
        write(tmp_path / "b.yaml", {"extends": "a.yaml", "section": {"label": "b"}})
        child = write(tmp_path / "c.yaml", {"extends": "b.yaml", "section": {"size": 3}})
        assert resolve_extends(child) == {"section": {"size": 3, "label": "b"}}

    def test_extends_is_relative_to_the_file_that_declares_it(self, tmp_path: Path) -> None:
        (tmp_path / "nested").mkdir()
        write(tmp_path / "base.yaml", {"section": {"label": "base"}})
        child = write(tmp_path / "nested" / "child.yaml", {"extends": "../base.yaml"})
        assert resolve_extends(child) == {"section": {"label": "base"}}

    def test_a_cycle_raises_instead_of_recursing_forever(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", {"extends": "b.yaml"})
        write(tmp_path / "b.yaml", {"extends": "a.yaml"})
        with pytest.raises(ValueError, match="cycle"):
            resolve_extends(tmp_path / "a.yaml")

    def test_the_same_parent_twice_is_not_a_cycle(self, tmp_path: Path) -> None:
        """A diamond is legal. Only a file extending its own ancestor is not."""
        write(tmp_path / "root.yaml", {"section": {"label": "root"}})
        write(tmp_path / "a.yaml", {"extends": "root.yaml", "section": {"size": 2}})
        write(tmp_path / "b.yaml", {"extends": "root.yaml"})
        child = write(tmp_path / "child.yaml", {"extends": ["a.yaml", "b.yaml"]})
        assert resolve_extends(child) == {"section": {"label": "root", "size": 2}}

    def test_a_non_path_extends_value_is_rejected(self, tmp_path: Path) -> None:
        child = write(tmp_path / "child.yaml", {"extends": 7})
        with pytest.raises(ValueError, match="path or list of paths"):
            resolve_extends(child)

    def test_extends_is_stripped_from_the_result(self, tmp_path: Path) -> None:
        write(tmp_path / "base.yaml", {})
        child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "section": {"size": 4}})
        assert "extends" not in resolve_extends(child)


class TestDottedOverrides:
    def test_values_are_parsed_as_yaml_not_kept_as_strings(self) -> None:
        assert parse_dotted_override("runtime.steps=3") == {"runtime": {"steps": 3}}
        assert parse_dotted_override("runtime.amp=false") == {"runtime": {"amp": False}}
        assert parse_dotted_override("runtime.name=null") == {"runtime": {"name": None}}
        assert parse_dotted_override("optimization.lr=1.0e-4") == {"optimization": {"lr": 1e-4}}

    def test_pyyaml_leaves_dotless_exponents_as_strings(self) -> None:
        """A YAML 1.1 quirk: `1e-4` needs a decimal point to parse as a float.

        The strict models coerce it back, so `--set optimization.lr=1e-4` still
        works. Without that coercion it would put a `str` where a float belongs.
        """
        assert parse_dotted_override("optimization.lr=1e-4") == {"optimization": {"lr": "1e-4"}}

    def test_a_value_containing_equals_keeps_its_tail(self) -> None:
        assert parse_dotted_override("runtime.note=a=b") == {"runtime": {"note": "a=b"}}

    def test_more_than_two_segments_nest(self) -> None:
        assert parse_dotted_override("a.b.c=1") == {"a": {"b": {"c": 1}}}

    @pytest.mark.parametrize("item", ["dataset.batch_size", "batch_size=8", ".batch_size=8", "=8"])
    def test_malformed_overrides_are_rejected(self, item: str) -> None:
        with pytest.raises(ValueError, match="Malformed --set"):
            parse_dotted_override(item)

    def test_overrides_apply_in_order(self) -> None:
        result = apply_overrides({}, ["s.size=1", "s.size=2"])
        assert result == {"s": {"size": 2}}

    def test_an_override_keeps_its_siblings(self) -> None:
        result = apply_overrides({"s": {"size": 1, "label": "keep"}}, ["s.size=9"])
        assert result == {"s": {"size": 9, "label": "keep"}}


class TestLoadConfig:
    def test_overrides_beat_both_the_child_and_its_parent(self, tmp_path: Path) -> None:
        write(tmp_path / "base.yaml", {"section": {"size": 1}})
        child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "section": {"size": 2}})
        assert load_config(Root, child, ["section.size=3"]).section.size == 3

    def test_an_unknown_key_names_its_own_path(self, tmp_path: Path) -> None:
        child = write(tmp_path / "child.yaml", {"section": {"nope": 1}})
        with pytest.raises(ValueError, match=r"section\.nope"):
            load_config(Root, child)

    def test_a_dotless_exponent_still_reaches_the_model_as_a_float(self, tmp_path: Path) -> None:
        child = write(tmp_path / "child.yaml", {})
        assert load_config(Root, child, ["section.rate=1e-4"]).section.rate == 1e-4

    def test_a_wrongly_typed_value_names_its_own_path(self, tmp_path: Path) -> None:
        child = write(tmp_path / "child.yaml", {"section": {"size": "big"}})
        with pytest.raises(ValueError, match=r"section\.size"):
            load_config(Root, child)


class TestStrictModel:
    def test_assigning_an_unknown_attribute_raises(self) -> None:
        with pytest.raises(ValueError, match="nope"):
            Section().nope = 1  # type: ignore[attr-defined]

    def test_assigning_a_wrong_type_raises_rather_than_sticking(self) -> None:
        section = Section()
        with pytest.raises(ValueError, match="size"):
            section.size = "big"  # type: ignore[assignment]
        assert section.size == 1

    def test_nested_defaults_are_not_shared_between_instances(self) -> None:
        first, second = Root(), Root()
        first.section.size = 99
        assert second.section.size == 1

    def test_to_dict_is_plain_nested_data(self) -> None:
        assert Root().to_dict() == {"section": {"size": 1, "rate": 0.1, "label": "none"}}
