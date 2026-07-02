"""Tests for the ChemBERTa-augmented CellCLIP path."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cellclip.training.config import CellCLIPModelConfig, load_training_config
from cellclip.training.dataset import build_tokenized_collate_fn
from cellclip.training.model import CellCLIPChemBERTa, CellCLIPChemBERTaFiLM
from morphoclip.data.metadata import MetadataIndex
from tests.cellclip.conftest import (
    DummyTokenizer,
    FakeTextModel,
    write_feature,
)


class FakeChemBERTaModel(torch.nn.Module):
    """RoBERTa-like stub exposing last_hidden_state and encoder layers."""

    def __init__(self, hidden_size: int = 32, layers: int = 3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(4096, hidden_size)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )
        self.encoder = SimpleNamespace(layer=self.layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_build_tokenized_collate_fn_adds_smiles_tokens_for_chemberta_variant(
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
    collate = build_tokenized_collate_fn(
        DummyTokenizer(),
        context_length=8,
        plate_contexts={"BR00116991": SimpleNamespace(cell_type="A549")},
        include_smiles_in_prompt=False,
        smiles_tokenizer=DummyTokenizer(),
        chemberta_context_length=6,
    )
    batch = collate([ds[0]])

    assert batch["text_tokens"]["input_ids"].shape == (1, 8)
    assert batch["smiles_tokens"]["input_ids"].shape == (1, 6)
    assert batch["has_smiles"].tolist() == [True]
    assert "SMILES:" not in batch["text"][0]


def test_build_tokenized_collate_fn_can_keep_smiles_in_prompt_with_chemberta(
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
    collate = build_tokenized_collate_fn(
        DummyTokenizer(),
        context_length=8,
        plate_contexts={"BR00116991": SimpleNamespace(cell_type="A549")},
        include_smiles_in_prompt=True,
        smiles_tokenizer=DummyTokenizer(),
        chemberta_context_length=6,
    )
    batch = collate([ds[0]])

    assert batch["smiles_tokens"]["input_ids"].shape == (1, 6)
    assert batch["has_smiles"].tolist() == [True]
    assert "SMILES:" in batch["text"][0]


def test_chemberta_variant_freezes_smiles_encoder_and_skips_film_without_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_loader(name: str, *_args, **_kwargs):
        if name == "fake-chem":
            return FakeChemBERTaModel(hidden_size=20)
        return FakeTextModel(hidden_size=24)

    monkeypatch.setattr("cellclip.training.model.AutoModel.from_pretrained", _fake_loader)

    model = CellCLIPChemBERTaFiLM(
        CellCLIPModelConfig(
            variant="chemberta_film",
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake-text",
            tokenizer_name="fake-text",
            chemberta_model_name="fake-chem",
            chemberta_tokenizer_name="fake-chem",
            freeze_chemberta=True,
        )
    )

    assert all(not param.requires_grad for param in model.chemberta.parameters())
    model.train()
    assert model.chemberta.training is False

    text_tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    smiles_tokens = {
        "input_ids": torch.randint(0, 100, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
    }

    prompt_hidden = model._encode_prompt_hidden(text_tokens)
    encoded_without_smiles = model.encode_text(
        text_tokens,
        smiles=smiles_tokens,
        has_smiles=torch.tensor([False, False]),
    )
    expected_without_smiles = model.text_proj(prompt_hidden)
    assert torch.allclose(encoded_without_smiles, expected_without_smiles)

    with torch.no_grad():
        model.film[-1].bias.fill_(0.1)

    encoded_with_smiles = model.encode_text(
        text_tokens,
        smiles=smiles_tokens,
        has_smiles=torch.tensor([True, True]),
    )
    encoded_with_smiles.sum().backward()

    assert model.text_proj.weight.grad is not None
    assert any(param.grad is not None for param in model.text.parameters())
    assert all(param.grad is None for param in model.chemberta.parameters())
    assert not torch.allclose(encoded_with_smiles, expected_without_smiles)


@pytest.mark.parametrize("fusion_type", ["film", "residual_add", "concat_mlp"])
def test_chemberta_variant_supports_multiple_fusion_modes(
    monkeypatch: pytest.MonkeyPatch,
    fusion_type: str,
) -> None:
    def _fake_loader(name: str, *_args, **_kwargs):
        if name == "fake-chem":
            return FakeChemBERTaModel(hidden_size=20)
        return FakeTextModel(hidden_size=24)

    monkeypatch.setattr("cellclip.training.model.AutoModel.from_pretrained", _fake_loader)

    model = CellCLIPChemBERTa(
        CellCLIPModelConfig(
            variant="chemberta",
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake-text",
            tokenizer_name="fake-text",
            chemberta_model_name="fake-chem",
            chemberta_tokenizer_name="fake-chem",
            chem_fusion_type=fusion_type,
            freeze_chemberta=True,
        )
    )
    text_tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    smiles_tokens = {
        "input_ids": torch.randint(0, 100, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
    }

    expected_without_smiles, empty_diag = model.encode_text_with_diagnostics(
        text_tokens,
        smiles=smiles_tokens,
        has_smiles=torch.tensor([False, False]),
    )
    assert empty_diag["fusion_delta_norm"] == pytest.approx(0.0)

    with torch.no_grad():
        model.chem_fusion[-1].bias.fill_(0.1)

    encoded_with_smiles, diagnostics = model.encode_text_with_diagnostics(
        text_tokens,
        smiles=smiles_tokens,
        has_smiles=torch.tensor([True, True]),
    )

    assert diagnostics["chem_hidden_norm"] > 0.0
    assert diagnostics["fusion_delta_norm"] > 0.0
    if fusion_type == "film":
        assert diagnostics["chem_gamma_norm"] > 0.0
        assert diagnostics["chem_beta_norm"] > 0.0
    assert not torch.allclose(encoded_with_smiles, expected_without_smiles)


def test_chemberta_variant_can_unfreeze_top_encoder_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_loader(name: str, *_args, **_kwargs):
        if name == "fake-chem":
            return FakeChemBERTaModel(hidden_size=20, layers=3)
        return FakeTextModel(hidden_size=24)

    monkeypatch.setattr("cellclip.training.model.AutoModel.from_pretrained", _fake_loader)

    model = CellCLIPChemBERTa(
        CellCLIPModelConfig(
            variant="chemberta",
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake-text",
            tokenizer_name="fake-text",
            chemberta_model_name="fake-chem",
            chemberta_tokenizer_name="fake-chem",
            chem_fusion_type="residual_add",
            freeze_chemberta=False,
            chemberta_tune_layers=1,
        )
    )

    frozen_params = list(model.chemberta.encoder.layer[0].parameters())
    tuned_params = list(model.chemberta.encoder.layer[-1].parameters())
    assert all(not param.requires_grad for param in model.chemberta.embedding.parameters())
    assert all(not param.requires_grad for param in frozen_params)
    assert all(param.requires_grad for param in tuned_params)

    model.train()
    assert model.chemberta.training is True

    text_tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    smiles_tokens = {
        "input_ids": torch.randint(0, 100, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
    }
    with torch.no_grad():
        model.chem_fusion[-1].bias.fill_(0.1)
    encoded_with_smiles = model.encode_text(
        text_tokens,
        smiles=smiles_tokens,
        has_smiles=torch.tensor([True, True]),
    )
    encoded_with_smiles.sum().backward()

    assert any(param.grad is not None for param in tuned_params)
    assert all(param.grad is None for param in frozen_params)


def test_chemberta_variant_loads_legacy_film_checkpoint_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_loader(name: str, *_args, **_kwargs):
        if name == "fake-chem":
            return FakeChemBERTaModel(hidden_size=20)
        return FakeTextModel(hidden_size=24)

    monkeypatch.setattr("cellclip.training.model.AutoModel.from_pretrained", _fake_loader)

    model = CellCLIPChemBERTaFiLM(
        CellCLIPModelConfig(
            variant="chemberta_film",
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake-text",
            tokenizer_name="fake-text",
            chemberta_model_name="fake-chem",
            chemberta_tokenizer_name="fake-chem",
        )
    )
    legacy_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("chem_fusion."):
            legacy_state[key.replace("chem_fusion.", "film.", 1)] = value
        else:
            legacy_state[key] = value

    loaded = model.load_state_dict(legacy_state, strict=True)
    assert not loaded.missing_keys
    assert not loaded.unexpected_keys


def test_load_training_config_accepts_chemberta_variant(tmp_path: Path) -> None:
    config_path = tmp_path / "chemberta.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                '  variant: "chemberta"',
                '  chemberta_model_name: "fake-chem"',
                '  chemberta_tokenizer_name: "fake-chem"',
                '  chem_fusion_type: "concat_mlp"',
                '  chem_prompt_policy: "keep_smiles"',
                "  freeze_chemberta: false",
                "  chemberta_tune_layers: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.model.variant == "chemberta"
    assert config.model.chemberta_model_name == "fake-chem"
    assert config.model.chem_fusion_type == "concat_mlp"
    assert config.model.chem_prompt_policy == "keep_smiles"
    assert config.model.freeze_chemberta is False
    assert config.model.chemberta_tune_layers == 2
