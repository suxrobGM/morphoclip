"""Local CellCLIP model for training on precomputed site features."""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from cellclip.benchmark.model import CrossChannelFormer
from cellclip.training.config import CellCLIPModelConfig


class MILPooling(nn.Module):
    """Channel-independent multi-instance pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        if pooling == "attention":
            self.V = nn.Linear(input_dim, hidden_dim)
            self.U = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool site bags from ``(B, M, C, D)`` to ``(B, C, D)``."""
        batch_size, num_sites, num_channels, width = x.shape

        if self.pooling == "attention":
            x = x.permute(0, 2, 1, 3).reshape(batch_size * num_channels, num_sites, width)
            h_v = torch.tanh(self.V(x))
            h_u = torch.sigmoid(self.U(x))
            attn_scores = self.attention(h_v * h_u)

            mask = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = torch.sum(attn_weights * x, dim=1)
            return pooled.view(batch_size, num_channels, width)

        if self.pooling == "median":
            return torch.median(x, dim=1).values

        return torch.mean(x, dim=1)


class CellCLIP(nn.Module):
    """CellCLIP image-text contrastive model."""

    def __init__(self, config: CellCLIPModelConfig):
        super().__init__()
        self.config = config
        self.context_length = config.context_length
        self.use_bias = config.use_bias

        self.visual = CrossChannelFormer(
            embed_dim=config.vision_width,
            layers=config.vision_layers,
            heads=config.vision_heads,
            output_dim=config.embed_dim,
            input_channels=config.input_channels,
        )
        self.image_pool = MILPooling(
            input_dim=config.vision_width,
            pooling=config.pooling,
        )
        self.text = AutoModel.from_pretrained(config.text_model_name)
        self.text_proj = nn.Linear(self.text.config.hidden_size, config.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if config.use_bias:
            self.bias = nn.Parameter(torch.ones([]) * -10)

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def encode_mil(self, image: torch.Tensor) -> torch.Tensor:
        """Pool site bags before the visual transformer."""
        return self.image_pool(image)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode pooled image bags."""
        return self.visual(image).to(self.dtype)

    def _encode_prompt_hidden(self, text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode prompt tokens before the final projection."""
        outputs = self.text(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
        )
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled.to(self.dtype)

    def encode_text(
        self,
        text: dict[str, torch.Tensor],
        *,
        smiles: dict[str, torch.Tensor] | None = None,
        has_smiles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokenized text prompts."""
        del smiles, has_smiles
        return self.text_proj(self._encode_prompt_hidden(text))

    def forward(
        self,
        image: torch.Tensor,
        text: dict[str, torch.Tensor],
        *,
        smiles: dict[str, torch.Tensor] | None = None,
        has_smiles: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        image_features = F.normalize(self.encode_image(image), dim=1)
        text_features = F.normalize(
            self.encode_text(text, smiles=smiles, has_smiles=has_smiles),
            dim=1,
        )

        if self.use_bias:
            return image_features, text_features, self.logit_scale, self.bias
        return image_features, text_features, self.logit_scale


class CellCLIPChemBERTa(CellCLIP):
    """CellCLIP variant that conditions prompt embeddings with ChemBERTa."""

    def __init__(self, config: CellCLIPModelConfig):
        super().__init__(config)
        prompt_width = self.text.config.hidden_size
        self.chemberta = AutoModel.from_pretrained(config.chemberta_model_name)
        self.chemberta_proj = nn.Linear(
            self.chemberta.config.hidden_size,
            prompt_width,
        )
        if config.chem_fusion_type == "film":
            self.chem_fusion = nn.Sequential(
                nn.LayerNorm(prompt_width),
                nn.Linear(prompt_width, prompt_width * 2),
            )
            self.film = self.chem_fusion
        elif config.chem_fusion_type == "residual_add":
            self.chem_fusion = nn.Sequential(
                nn.LayerNorm(prompt_width),
                nn.Linear(prompt_width, prompt_width),
            )
        else:
            self.chem_fusion = nn.Sequential(
                nn.LayerNorm(prompt_width * 2),
                nn.Linear(prompt_width * 2, prompt_width * 2),
                nn.GELU(),
                nn.Linear(prompt_width * 2, prompt_width),
            )
        nn.init.zeros_(self.chem_fusion[-1].weight)
        nn.init.zeros_(self.chem_fusion[-1].bias)
        self._configure_chemberta_trainability()

    def _configure_chemberta_trainability(self) -> None:
        if self.config.freeze_chemberta or self.config.chemberta_tune_layers > 0:
            for param in self.chemberta.parameters():
                param.requires_grad = False
        if self.config.chemberta_tune_layers > 0:
            layers = getattr(getattr(self.chemberta, "encoder", None), "layer", None)
            if layers is None:
                raise ValueError("ChemBERTa encoder layers are unavailable for partial tuning")
            tune_count = min(self.config.chemberta_tune_layers, len(layers))
            for layer in layers[-tune_count:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif not self.config.freeze_chemberta:
            for param in self.chemberta.parameters():
                param.requires_grad = True
        if self.config.freeze_chemberta:
            self.chemberta.eval()

    def train(self, mode: bool = True) -> CellCLIPChemBERTa:
        """Keep ChemBERTa in eval mode when fully frozen."""
        super().train(mode)
        if self.config.freeze_chemberta:
            self.chemberta.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        """Support legacy FiLM checkpoints after the fusion-module rename."""
        if "chem_fusion.0.weight" not in state_dict and "film.0.weight" in state_dict:
            remapped = dict(state_dict)
            for key, value in list(state_dict.items()):
                if key.startswith("film."):
                    remapped[key.replace("film.", "chem_fusion.", 1)] = value
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _masked_mean_norm(values: torch.Tensor, mask: torch.Tensor) -> float:
        if mask.numel() == 0 or not bool(mask.any()):
            return 0.0
        selected = values[mask]
        return float(selected.float().norm(dim=1).mean().item())

    def _encode_smiles_hidden(self, smiles: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode SMILES tokens with configurable pooling."""
        context = torch.no_grad() if self.config.freeze_chemberta else nullcontext()
        with context:
            outputs = self.chemberta(
                input_ids=smiles["input_ids"],
                attention_mask=smiles["attention_mask"],
            )
        hidden = outputs.last_hidden_state
        if self.config.chemberta_pooling == "cls":
            pooled = hidden[:, 0, :]
            return self.chemberta_proj(pooled.to(self.dtype))
        mask = smiles["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.chemberta_proj(pooled.to(self.dtype))

    def _fuse_prompt_and_smiles(
        self,
        prompt_hidden: torch.Tensor,
        smiles_hidden: torch.Tensor,
        has_smiles: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = has_smiles.to(device=prompt_hidden.device, dtype=torch.bool)
        if self.config.chem_fusion_type == "film":
            gamma, beta = self.chem_fusion(smiles_hidden).chunk(2, dim=-1)
            delta = gamma * prompt_hidden + beta
            diagnostics = {
                "chem_gamma_norm": self._masked_mean_norm(gamma, mask),
                "chem_beta_norm": self._masked_mean_norm(beta, mask),
            }
        elif self.config.chem_fusion_type == "residual_add":
            delta = self.chem_fusion(smiles_hidden)
            diagnostics = {}
        else:
            delta = self.chem_fusion(torch.cat([prompt_hidden, smiles_hidden], dim=-1))
            diagnostics = {}
        fused_hidden = prompt_hidden + delta
        fused_hidden = torch.where(mask.unsqueeze(1), fused_hidden, prompt_hidden)
        diagnostics.update(
            {
                "chem_hidden_norm": self._masked_mean_norm(smiles_hidden, mask),
                "fusion_delta_norm": self._masked_mean_norm(delta, mask),
            }
        )
        return fused_hidden, diagnostics

    def encode_text_with_diagnostics(
        self,
        text: dict[str, torch.Tensor],
        *,
        smiles: dict[str, torch.Tensor] | None = None,
        has_smiles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Encode prompt text and return optional ChemBERTa diagnostics."""
        prompt_hidden = self._encode_prompt_hidden(text)
        if smiles is None or has_smiles is None:
            return self.text_proj(prompt_hidden), {}
        fused_hidden, diagnostics = self._fuse_prompt_and_smiles(
            prompt_hidden,
            self._encode_smiles_hidden(smiles),
            has_smiles,
        )
        return self.text_proj(fused_hidden), diagnostics

    def encode_text(
        self,
        text: dict[str, torch.Tensor],
        *,
        smiles: dict[str, torch.Tensor] | None = None,
        has_smiles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse prompt text with ChemBERTa conditioning."""
        text_features, _ = self.encode_text_with_diagnostics(
            text,
            smiles=smiles,
            has_smiles=has_smiles,
        )
        return text_features


class CellCLIPChemBERTaFiLM(CellCLIPChemBERTa):
    """Backward-compatible alias for the FiLM ChemBERTa variant."""


def build_cellclip_model(config: CellCLIPModelConfig) -> CellCLIP:
    """Instantiate the requested CellCLIP model variant."""
    if config.variant == "chemberta_film":
        return CellCLIPChemBERTaFiLM(config)
    if config.variant == "chemberta":
        return CellCLIPChemBERTa(config)
    return CellCLIP(config)
