"""Tests for the local CellCLIP training pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import benchmark.splits as splits_module
from cellclip.benchmark.checkpoint import load_cellclip_visual_encoder
from cellclip.training.config import (
    CellCLIPDatasetConfig,
    CellCLIPModelConfig,
    load_training_config,
)
from cellclip.training.dataset import (
    build_tokenized_collate_fn,
    prepare_datasets,
)
from cellclip.training.model import CellCLIP
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationType
from tests.cellclip.conftest import (
    HIDDEN_DIM,
    NUM_CHANNELS,
    DummyTokenizer,
    FakeTextModel,
    write_feature,
)


@pytest.fixture
def official_split_metadata_csv(tmp_path: Path) -> Path:
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
                "BR00117020,A01,ccsbBroad304_00900,KCNN1,A549,ORF,48,low,TRUE,2",
                "BR00116991,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,24,low,TRUE,3",
                "BR00117017,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,48,high,TRUE,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_build_tokenized_collate_fn_adds_bert_tokens(
    tmp_path: Path,
    metadata_index: MetadataIndex,
) -> None:
    write_feature(tmp_path, "BR00116991", "A01", sites=2)
    from morphoclip.data.dataset import MorphoCLIPDataset

    ds = MorphoCLIPDataset(
        feature_dir=tmp_path,
        metadata=metadata_index,
        plates=["BR00116991"],
    )
    plate_contexts = {"BR00116991": SimpleNamespace(cell_type="A549")}
    collate = build_tokenized_collate_fn(
        DummyTokenizer(),
        context_length=8,
        plate_contexts=plate_contexts,
    )
    batch = collate([ds[0]])

    assert batch["features"].shape == (1, 2, NUM_CHANNELS, HIDDEN_DIM)
    assert batch["text_tokens"]["input_ids"].shape == (1, 8)
    assert batch["text_tokens"]["attention_mask"].shape == (1, 8)
    assert batch["text"][0].startswith("A549 cells treated with compound:")
    assert "smiles_tokens" not in batch


def test_prepare_datasets_uses_official_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    official_split_metadata_csv: Path,
) -> None:
    monkeypatch.setattr(
        splits_module,
        "OFFICIAL_SPLIT_METADATA_PATH",
        official_split_metadata_csv,
    )
    write_feature(tmp_path, "BR00117000", "A04", sites=2)
    write_feature(tmp_path, "BR00117020", "A01", sites=1)
    write_feature(tmp_path, "BR00116991", "A01", sites=2)
    write_feature(tmp_path, "BR00117017", "A01", sites=1)

    dataset_cfg = CellCLIPDatasetConfig(
        dataset_config_path="configs/dataset.yml",
        feature_root=str(tmp_path),
        split_strategy="cpjump1_official_representation",
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    model_cfg = CellCLIPModelConfig(tokenizer_name="bert-base-cased")

    from unittest.mock import patch

    with (
        patch(
            "cellclip.training.dataset.AutoTokenizer.from_pretrained",
            return_value=DummyTokenizer(),
        ),
        patch(
            "cellclip.training.dataset.benchmark_splits_module.load_plate_contexts",
            return_value={
                "BR00117000": SimpleNamespace(cell_type="U2OS"),
                "BR00117020": SimpleNamespace(cell_type="A549"),
                "BR00116991": SimpleNamespace(cell_type="A549"),
                "BR00117017": SimpleNamespace(cell_type="A549"),
            },
        ),
    ):
        prepared = prepare_datasets(dataset_cfg, model_cfg)

    assert len(prepared.train_dataset) == 2
    assert len(prepared.eval_dataset) == 1
    train_batch = next(iter(prepared.train_loader))
    assert train_batch["features"].shape[2:] == (NUM_CHANNELS, HIDDEN_DIM)
    assert all("cells treated with" in prompt for prompt in train_batch["text"])


def test_prepare_datasets_unique_perturbations_preserve_timepoint_distinct_wells(
    tmp_path: Path,
    metadata_index: MetadataIndex,
) -> None:
    write_feature(tmp_path, "BR00116991", "A01", sites=2)
    write_feature(tmp_path, "BR00117017", "A01", sites=1)
    write_feature(tmp_path, "BR00117000", "A04", sites=1)
    split_manifest = tmp_path / "split_manifest.csv"
    split_manifest.write_text(
        "\n".join(
            [
                "Metadata_Plate,Metadata_Well,subset",
                "BR00116991,A01,train",
                "BR00117017,A01,train",
                "BR00117000,A04,train",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_cfg = CellCLIPDatasetConfig(
        dataset_config_path="configs/dataset.yml",
        feature_root=str(tmp_path),
        split_strategy="cellclip_cpjump_style",
        split_manifest_path=str(split_manifest),
        subset="train",
        eval_subset="train",
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    model_cfg = CellCLIPModelConfig(tokenizer_name="bert-base-cased")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "cellclip.training.dataset.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: DummyTokenizer(),
        )
        monkeypatch.setattr(
            "cellclip.training.dataset.benchmark_splits_module.load_plate_contexts",
            lambda: {
                "BR00116991": SimpleNamespace(cell_type="A549"),
                "BR00117017": SimpleNamespace(cell_type="A549"),
                "BR00117000": SimpleNamespace(cell_type="U2OS"),
            },
        )
        assert len(prepare_datasets(dataset_cfg, model_cfg).train_dataset) == 3
        dataset_cfg.unique_perturbations = True
        assert len(prepare_datasets(dataset_cfg, model_cfg).train_dataset) == 3


def test_prepare_datasets_supports_split_specific_site_caps(
    tmp_path: Path,
    metadata_index: MetadataIndex,
) -> None:
    write_feature(tmp_path, "BR00116991", "A01", sites=3)
    write_feature(tmp_path, "BR00116991", "A03", sites=3)
    split_manifest = tmp_path / "split_manifest.csv"
    split_manifest.write_text(
        "\n".join(
            [
                "Metadata_Plate,Metadata_Well,subset",
                "BR00116991,A01,train",
                "BR00116991,A03,test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_cfg = CellCLIPDatasetConfig(
        dataset_config_path="configs/dataset.yml",
        feature_root=str(tmp_path),
        split_strategy="cellclip_cpjump_style",
        split_manifest_path=str(split_manifest),
        subset="train",
        eval_subset="test",
        train_max_sites_per_well=1,
        eval_max_sites_per_well=None,
        batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    model_cfg = CellCLIPModelConfig(tokenizer_name="bert-base-cased")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "cellclip.training.dataset.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: DummyTokenizer(),
        )
        monkeypatch.setattr(
            "cellclip.training.dataset.benchmark_splits_module.load_plate_contexts",
            lambda: {"BR00116991": SimpleNamespace(cell_type="A549")},
        )
        prepared = prepare_datasets(dataset_cfg, model_cfg)

    train_batch = next(iter(prepared.train_loader))
    eval_batch = next(iter(prepared.eval_loader))
    assert train_batch["features"].shape[1] == 1
    assert eval_batch["features"].shape[1] == 3


def test_build_tokenized_collate_fn_uses_upstream_style_crispr_prompt() -> None:
    collate = build_tokenized_collate_fn(
        DummyTokenizer(),
        context_length=16,
        plate_contexts={"BR00117000": SimpleNamespace(cell_type="U2OS")},
    )
    sample = SimpleNamespace(
        features=torch.randn(2, NUM_CHANNELS, HIDDEN_DIM),
        plate="BR00117000",
        well="A04",
        text="ignored",
        pert_info=SimpleNamespace(
            pert_type=PerturbationType.CRISPR,
            pert_iname="",
            broad_sample="BRDN0000259015",
            smiles="",
            gene="OPRL1",
            target_sequence="ACGTACGT",
            control_type="",
            negcon_control_type="",
        ),
    )
    batch = collate([sample])
    assert (
        batch["text"][0]
        == "U2OS cells treated with crispr sequence: ACGTACGT, targeting genes: OPRL1"
    )


def test_training_checkpoint_is_visual_loader_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cellclip.training.model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeTextModel(hidden_size=24),
    )

    model = CellCLIP(
        CellCLIPModelConfig(
            embed_dim=512,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake",
            tokenizer_name="fake",
        )
    )
    checkpoint_path = tmp_path / "train_ckpt.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    loaded = load_cellclip_visual_encoder(
        model_path=str(checkpoint_path),
        device="cpu",
        input_dim=16,
        embed_dim=512,
        vision_layers=2,
        vision_heads=4,
        input_channels=5,
    )

    sample = torch.randn(2, 5, 16)
    expected = model.encode_image(sample)
    actual = loaded.encode_image(sample)
    assert torch.allclose(actual, expected)

    bag = torch.randn(2, 3, 5, 16)
    expected_bag = model.encode_image(model.encode_mil(bag))
    actual_bag = loaded.encode_image(loaded.encode_mil(bag))
    assert torch.allclose(actual_bag, expected_bag)


def test_cellclip_text_encoder_is_trainable_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cellclip.training.model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeTextModel(hidden_size=24),
    )

    model = CellCLIP(
        CellCLIPModelConfig(
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake",
            tokenizer_name="fake",
        )
    )

    assert all(param.requires_grad for param in model.text.parameters())

    model.train()
    assert model.training is True
    assert model.text.training is True

    text_tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    text_features = model.encode_text(text_tokens)
    text_features.sum().backward()

    assert model.text_proj.weight.grad is not None
    assert any(param.grad is not None for param in model.text.parameters())


def test_load_training_config_supports_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "dataset:",
                '  feature_root: "data/features_cellclip_base"',
                "  batch_size: 16",
                "optimization:",
                "  lr: 5.0e-4",
                "  warmup_steps: 100",
                "runtime:",
                '  run_name: "base_run"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "\n".join(
            [
                'extends: "base.yaml"',
                "dataset:",
                "  batch_size: 64",
                "optimization:",
                "  warmup_steps: 1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_training_config(child)

    assert config.dataset.feature_root == "data/features_cellclip_base"
    assert config.dataset.batch_size == 64
    assert config.optimization.lr == pytest.approx(5.0e-4)
    assert config.optimization.warmup_steps == 1000
    assert config.runtime.run_name == "base_run"
