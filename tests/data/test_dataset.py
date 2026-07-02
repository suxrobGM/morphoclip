"""Tests for morphoclip.data.dataset and morphoclip.data.splits modules.

Uses real metadata from data/metadata/ and synthesized .pt feature files.
"""

from pathlib import Path

import pytest
import torch

import benchmark.split_contexts as split_contexts_module
from benchmark.splits import (
    build_split_groups as benchmark_build_split_groups,
)
from benchmark.splits import (
    build_split_manifest as benchmark_build_split_manifest,
)
from benchmark.splits import (
    create_splits as benchmark_create_splits,
)
from morphoclip.data.dataset import (
    MorphoCLIPDataset,
    MorphoCLIPSample,
    collate_fn,
)
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationType
from morphoclip.data.splits import create_splits

BATCH = "2020_11_04_CPJUMP1"
HIDDEN_DIM = 384
NUM_CHANNELS = 5


@pytest.fixture
def metadata_index(metadata_dir: Path) -> MetadataIndex:
    """Build MetadataIndex from real metadata."""
    return MetadataIndex.from_directory(metadata_dir, batch=BATCH)


@pytest.fixture
def fake_features(tmp_path: Path) -> Path:
    """Create synthetic feature .pt files mimicking extracted features.

    Creates features for plate BR00116991 (compound plate):
    - 4 sites across 2 wells (A01 has fields f01,f02; A03 has fields f01,f02)
    """
    plate_dir = tmp_path / "BR00116991"
    plate_dir.mkdir()

    for row, col in [(1, 1), (1, 3)]:  # A01 and A03
        for field in [1, 2]:
            feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
            filename = f"r{row:02d}c{col:02d}f{field:02d}.pt"
            torch.save(feat, plate_dir / filename)

    return tmp_path


class TestMorphoCLIPDataset:
    def test_basic_construction(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        # 2 wells: A01 and A03
        assert len(ds) == 2

    def test_getitem_returns_sample(
        self, fake_features: Path, metadata_index: MetadataIndex
    ) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        sample = ds[0]
        assert isinstance(sample, MorphoCLIPSample)
        # 2 sites per well, 5 channels, 384 dim
        assert sample.features.shape == (2, NUM_CHANNELS, HIDDEN_DIM)
        assert isinstance(sample.text, str)
        assert len(sample.text) > 0

    def test_exclude_controls(
        self, fake_features: Path, metadata_index: MetadataIndex, tmp_path: Path
    ) -> None:
        """A02 is DMSO (negcon). Add features for it and verify exclusion."""
        plate_dir = tmp_path / "BR00116991"
        feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
        torch.save(feat, plate_dir / "r01c02f01.pt")  # A02

        ds_all = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
            exclude_controls=False,
        )
        ds_no_ctrl = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
            exclude_controls=True,
        )
        assert len(ds_all) > len(ds_no_ctrl)

    def test_pert_type_filter(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
            pert_types={PerturbationType.COMPOUND},
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample.pert_info.pert_type == PerturbationType.COMPOUND

    def test_metadata_property(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        assert ds.metadata is metadata_index

    def test_text_levels(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        for level in ["name_only", "name_target", "full"]:
            ds = MorphoCLIPDataset(
                feature_dir=fake_features,
                metadata=metadata_index,
                plates=["BR00116991"],
                text_level=level,
            )
            sample = ds[0]
            assert isinstance(sample.text, str)
            assert len(sample.text) > 0

    def test_max_sites_per_well(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
            max_sites_per_well=1,
        )
        sample = ds[0]
        assert sample.features.shape[0] == 1

    def test_missing_plate_dir(self, tmp_path: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["NONEXISTENT"],
        )
        assert len(ds) == 0


class TestCollateFn:
    def test_pads_variable_sites(
        self, fake_features: Path, metadata_index: MetadataIndex, tmp_path: Path
    ) -> None:
        """Add a 3rd site to one well so sites differ across wells."""
        plate_dir = tmp_path / "BR00116991"
        feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
        torch.save(feat, plate_dir / "r01c01f03.pt")  # A01 now has 3 sites

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        batch = collate_fn([ds[0], ds[1]])
        assert batch["features"].shape[0] == 2  # batch size
        assert batch["features"].shape[1] == 3  # max sites (A01 has 3)
        assert batch["site_mask"].shape == (2, 3)
        # A01 has 3 sites, A03 has 2 — mask should differ
        assert batch["site_mask"][0].sum() == 3
        assert batch["site_mask"][1].sum() == 2


class TestCreateSplits:
    @pytest.fixture
    def experiment_metadata_tsv(self, tmp_path: Path) -> Path:
        path = tmp_path / "experiment-metadata.tsv"
        path.write_text(
            "\n".join(
                [
                    "Batch\tPlate_Map_Name\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116991\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116992\tcompound\tA549\t48",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116995\tcompound\tU2OS\t24",
                    "2020_11_04_CPJUMP1\tcrispr\tBR00117000\tcrispr\tA549\t96",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _index_by_plate_well(ds: MorphoCLIPDataset) -> dict[tuple[str, str], int]:
        return {(plate, well): i for i, (plate, well, _) in enumerate(ds.index_entries)}

    def _make_multi_well_dataset(
        self, tmp_path: Path, metadata_index: MetadataIndex
    ) -> MorphoCLIPDataset:
        """Create a dataset with enough wells for meaningful splits."""
        plate_dir = tmp_path / "BR00116991"
        plate_dir.mkdir(exist_ok=True)
        # Create features for 10 wells across first two rows
        for col in range(1, 11):
            feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
            torch.save(feat, plate_dir / f"r01c{col:02d}f01.pt")

        return MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
        )

    @staticmethod
    def _write_well_feature(feature_root: Path, plate: str, well: str, site: int = 1) -> None:
        row = ord(well[0].upper()) - ord("A") + 1
        col = int(well[1:])
        plate_dir = feature_root / plate
        plate_dir.mkdir(exist_ok=True)
        feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
        torch.save(feat, plate_dir / f"r{row:02d}c{col:02d}f{site:02d}.pt")

    @pytest.fixture
    def official_split_metadata_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "cpjump1_metadata.csv"
        path.write_text(
            "\n".join(
                [
                    (
                        "Metadata_Plate,Metadata_Well,Metadata_broad_sample,Metadata_target,"
                        "Metadata_cell_line,Metadata_experiment_type,Metadata_timepoint,"
                        "Metadata_timepoint_code,Metadata_target_is_across,Metadata_target_radix"
                    ),
                    "BR00117000,A04,BRDN0000259015,OPRL1,U2OS,CRISPR,144,high,TRUE,1",
                    "BR00117000,A11,BRDN0000259016,OPRL1,U2OS,CRISPR,144,high,TRUE,1",
                    "BR00116991,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,24,low,TRUE,2",
                    "BR00117017,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,48,high,TRUE,2",
                    "BR00117020,A01,ccsbBroad304_00900,KCNN1,A549,ORF,48,low,TRUE,3",
                    "BR00117003,A01,BRDN0001480888,HIF1A,A549,CRISPR,144,high,TRUE,4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_invalid_strategy(self, tmp_path: Path, metadata_index: MetadataIndex) -> None:
        ds = self._make_multi_well_dataset(tmp_path, metadata_index)
        with pytest.raises(ValueError, match="Unknown split strategy"):
            create_splits(ds, strategy="invalid")

    def test_deterministic(self, tmp_path: Path, metadata_index: MetadataIndex) -> None:
        for plate, well in [
            ("BR00117003", "A01"),
            ("BR00117020", "A01"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117003", "BR00117020", "BR00116991", "BR00117017"],
        )
        train1, val1, test1 = benchmark_create_splits(
            ds, strategy="cpjump1_official_representation"
        )
        train2, val2, test2 = benchmark_create_splits(
            ds, strategy="cpjump1_official_representation"
        )
        assert list(train1.indices) == list(train2.indices)
        assert list(val1.indices) == list(val2.indices)
        assert list(test1.indices) == list(test2.indices)

    def test_cpjump1_official_representation_split_assigns_expected_subsets(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        official_split_metadata_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            split_contexts_module,
            "OFFICIAL_SPLIT_METADATA_PATH",
            official_split_metadata_csv,
        )
        for plate, well in [
            ("BR00117003", "A01"),
            ("BR00117020", "A01"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117003", "BR00117020", "BR00116991", "BR00117017"],
        )
        index_map = self._index_by_plate_well(ds)

        train, val, test = benchmark_create_splits(ds, strategy="cpjump1_official_representation")

        assert set(train.indices) == {
            index_map[("BR00117003", "A01")],
            index_map[("BR00117020", "A01")],
        }
        assert set(val.indices) == {index_map[("BR00116991", "A01")]}
        assert set(test.indices) == {index_map[("BR00117017", "A01")]}

    def test_cpjump1_official_representation_groups_include_subset_bucket(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        official_split_metadata_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            split_contexts_module,
            "OFFICIAL_SPLIT_METADATA_PATH",
            official_split_metadata_csv,
        )
        for plate, well in [
            ("BR00117003", "A01"),
            ("BR00117020", "A01"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117003", "BR00117020", "BR00116991", "BR00117017"],
        )
        groups = benchmark_build_split_groups(ds, strategy="cpjump1_official_representation")
        assert any(key.startswith("train::") for key in groups)
        assert any(key.startswith("validate::") for key in groups)
        assert any(key.startswith("test::") for key in groups)

    def test_cpjump1_official_gene_compound_split_keeps_targets_together(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        official_split_metadata_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            split_contexts_module,
            "OFFICIAL_SPLIT_METADATA_PATH",
            official_split_metadata_csv,
        )
        for plate, well in [
            ("BR00117000", "A04"),
            ("BR00117000", "A11"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
            ("BR00117020", "A01"),
            ("BR00117003", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117000", "BR00116991", "BR00117017", "BR00117020", "BR00117003"],
        )
        index_map = self._index_by_plate_well(ds)

        train, val, test = benchmark_create_splits(ds, strategy="cpjump1_official_gene_compound")

        assert set(train.indices) == {
            index_map[("BR00117000", "A04")],
            index_map[("BR00117000", "A11")],
            index_map[("BR00116991", "A01")],
            index_map[("BR00117017", "A01")],
        }
        assert set(val.indices) == {index_map[("BR00117020", "A01")]}
        assert set(test.indices) == {index_map[("BR00117003", "A01")]}

    def test_cpjump1_official_gene_compound_groups_by_target(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        official_split_metadata_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            split_contexts_module,
            "OFFICIAL_SPLIT_METADATA_PATH",
            official_split_metadata_csv,
        )
        for plate, well in [
            ("BR00117000", "A04"),
            ("BR00117000", "A11"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117000", "BR00116991", "BR00117017"],
        )
        groups = benchmark_build_split_groups(ds, strategy="cpjump1_official_gene_compound")

        assert set(groups["OPRL1"]) == {0, 1}
        assert set(groups["CACNB4"]) == {2, 3}

    def test_cellclip_cpjump_style_split_groups_broad_samples_within_slice(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        experiment_metadata_tsv = tmp_path / "experiment-metadata.tsv"
        experiment_metadata_tsv.write_text(
            "\n".join(
                [
                    "Batch\tPlate_Map_Name\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116991\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116992\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116993\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116994\tcompound\tA549\t24",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(split_contexts_module, "METADATA_PATH", experiment_metadata_tsv)

        for plate, well in [
            ("BR00116991", "A01"),
            ("BR00116992", "A01"),
            ("BR00116993", "A03"),
            ("BR00116994", "A04"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991", "BR00116992", "BR00116993", "BR00116994"],
        )
        index_map = self._index_by_plate_well(ds)

        train, val, test = benchmark_create_splits(ds, strategy="cellclip_cpjump_style")

        assert len(val.indices) == 0
        assert len(train.indices) + len(test.indices) == len(ds)
        assert len(train.indices) > 0
        assert len(test.indices) > 0
        a01_indices = {
            index_map[("BR00116991", "A01")],
            index_map[("BR00116992", "A01")],
        }
        assert a01_indices.issubset(set(train.indices)) or a01_indices.issubset(set(test.indices))

    def test_cellclip_cpjump_style_groups_by_slice_and_broad_sample(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        experiment_metadata_tsv = tmp_path / "experiment-metadata.tsv"
        experiment_metadata_tsv.write_text(
            "\n".join(
                [
                    "Batch\tPlate_Map_Name\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116991\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116992\tcompound\tA549\t24",
                    "2020_11_04_CPJUMP1\tcompound\tBR00116995\tcompound\tU2OS\t24",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(split_contexts_module, "METADATA_PATH", experiment_metadata_tsv)

        for plate, well in [
            ("BR00116991", "A01"),
            ("BR00116992", "A01"),
            ("BR00116995", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991", "BR00116992", "BR00116995"],
        )
        groups = benchmark_build_split_groups(ds, strategy="cellclip_cpjump_style")

        assert groups["A549::compound::24::BRD-A86665761-001-01-1"] == [0, 1]
        assert groups["U2OS::compound::24::BRD-A86665761-001-01-1"] == [2]

    def test_build_split_manifest_uses_plate_well_keys(
        self,
        tmp_path: Path,
        metadata_index: MetadataIndex,
        official_split_metadata_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            split_contexts_module,
            "OFFICIAL_SPLIT_METADATA_PATH",
            official_split_metadata_csv,
        )
        for plate, well in [
            ("BR00117003", "A01"),
            ("BR00117020", "A01"),
            ("BR00116991", "A01"),
            ("BR00117017", "A01"),
        ]:
            self._write_well_feature(tmp_path, plate, well)

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00117003", "BR00117020", "BR00116991", "BR00117017"],
        )

        manifest = benchmark_build_split_manifest(ds, strategy="cpjump1_official_representation")

        assert manifest[["Metadata_Plate", "Metadata_Well"]].duplicated().sum() == 0
        assert set(manifest["subset"]) == {"train", "validate", "test"}
        test_row = manifest.query("Metadata_Plate=='BR00117017' and Metadata_Well=='A01'")
        assert test_row.iloc[0]["subset"] == "test"
        assert test_row.iloc[0]["Metadata_timepoint_code"] == "high"
