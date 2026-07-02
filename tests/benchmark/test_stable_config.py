"""Unit tests for benchmark.stable_config (pure; no copairs/sklearn needed)."""

from pathlib import Path

import pytest

from benchmark.stable_config import (
    TIMELINE_CHOICES,
    load_benchmark_config,
    normalize_timelines,
    resolve_params,
    resolve_path,
)


class TestNormalizeTimelines:
    def test_none_returns_all_choices(self):
        assert normalize_timelines(None) == list(TIMELINE_CHOICES)

    def test_single_string(self):
        assert normalize_timelines("short") == ["short"]

    def test_case_and_whitespace_normalized(self):
        assert normalize_timelines(["  LONG "]) == ["long"]

    def test_deduplicates_preserving_order(self):
        assert normalize_timelines(["long", "short", "long"]) == ["long", "short"]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid timeline"):
            normalize_timelines(["medium"])


class TestResolvePath:
    def test_none(self):
        assert resolve_path(Path("/root"), None) is None

    def test_absolute_kept(self):
        absolute = Path("/tmp/data").resolve()
        assert resolve_path(Path("/root"), str(absolute)) == absolute

    def test_relative_joined(self):
        assert resolve_path(Path("/root"), "sub/dir") == Path("/root") / "sub/dir"


class TestLoadBenchmarkConfig:
    def test_missing_returns_empty(self, tmp_path):
        assert load_benchmark_config(tmp_path / "nope.yml") == {}

    def test_extracts_benchmark_section(self, tmp_path):
        path = tmp_path / "b.yml"
        path.write_text("benchmark:\n  batch: X\n  test_mode: true\n", encoding="utf-8")
        assert load_benchmark_config(path) == {"batch": "X", "test_mode": True}

    def test_flat_mapping_returned_as_is(self, tmp_path):
        path = tmp_path / "b.yml"
        path.write_text("batch: Y\n", encoding="utf-8")
        assert load_benchmark_config(path) == {"batch": "Y"}

    def test_non_mapping_raises(self, tmp_path):
        path = tmp_path / "b.yml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_benchmark_config(path)


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "benchmark.yml"
    path.write_text(body, encoding="utf-8")
    return path


class TestResolveParams:
    def test_defaults_when_no_config(self, tmp_path):
        params = resolve_params(
            config=tmp_path / "absent.yml",
            profiles_dir=None,
            output_dir=None,
            batch=None,
            test_mode=None,
            cell_filter=None,
            batch_correction=None,
            pca_kernel=None,
            pca_n_components=None,
            split_manifest=None,
            split_subset=None,
            timelines=None,
        )
        assert params.batch == "2020_11_04_CPJUMP1"
        assert params.test_mode is False
        assert params.batch_correction is False
        assert params.pca_kernel == "linear"
        assert params.pca_n_components == 500
        assert params.timelines == list(TIMELINE_CHOICES)
        assert params.profiles_path.name == "profiles"
        assert params.figures_dir == params.output_path / "figures"
        assert params.tables_dir == params.output_path / "tables"
        assert params.manifest is None
        assert params.subset is None

    def test_cli_overrides_take_precedence_over_config(self, tmp_path):
        config = _write_config(
            tmp_path,
            "benchmark:\n  batch: CFG\n  pca_n_components: 10\n  timelines: [short]\n",
        )
        params = resolve_params(
            config=config,
            profiles_dir=None,
            output_dir=None,
            batch="CLI",
            test_mode=True,
            cell_filter=None,
            batch_correction=None,
            pca_kernel=None,
            pca_n_components=None,
            split_manifest=None,
            split_subset=None,
            timelines=None,
        )
        assert params.batch == "CLI"
        assert params.test_mode is True
        # config value used when no CLI override
        assert params.pca_n_components == 10
        assert params.timelines == ["short"]

    def test_config_used_when_no_override(self, tmp_path):
        config = _write_config(
            tmp_path, "benchmark:\n  batch_correction: true\n  pca_kernel: rbf\n"
        )
        params = resolve_params(
            config=config,
            profiles_dir=None,
            output_dir=None,
            batch=None,
            test_mode=None,
            cell_filter=None,
            batch_correction=None,
            pca_kernel=None,
            pca_n_components=None,
            split_manifest=None,
            split_subset=None,
            timelines=None,
        )
        assert params.batch_correction is True
        assert params.pca_kernel == "rbf"

    def test_split_subset_without_manifest_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires --split-manifest"):
            resolve_params(
                config=tmp_path / "absent.yml",
                profiles_dir=None,
                output_dir=None,
                batch=None,
                test_mode=None,
                cell_filter=None,
                batch_correction=None,
                pca_kernel=None,
                pca_n_components=None,
                split_manifest=None,
                split_subset="train",
                timelines=None,
            )

    def test_manifest_without_subset_raises(self, tmp_path):
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="--split-subset is required"):
            resolve_params(
                config=tmp_path / "absent.yml",
                profiles_dir=None,
                output_dir=None,
                batch=None,
                test_mode=None,
                cell_filter=None,
                batch_correction=None,
                pca_kernel=None,
                pca_n_components=None,
                split_manifest=str(manifest),
                split_subset=None,
                timelines=None,
            )
