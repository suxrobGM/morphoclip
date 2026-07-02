"""MorphoCLIP text encoder: metadata -> BioClinical ModernBERT -> projection.

Encodes perturbation metadata into dense embeddings using a frozen
BioClinical ModernBERT backbone and a trainable projection head.

Architecture::

    metadata dict
        -> build_prompts() (constructs natural language descriptions)
        -> BioClinical ModernBERT (frozen, 150M params, 8192 context)
        -> [CLS] pooling (768-d)
        -> ProjectionHead (768 -> 512, LayerNorm + GELU + Dropout)
        -> L2 normalize
        -> text embedding t in R^512
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from morphoclip.models.projection_head import ProjectionHead
from morphoclip.models.prompts import TEMPLATES, build_prompts


class MorphoCLIPTextEncoder(nn.Module):
    """Complete text encoder for MorphoCLIP.

    Takes perturbation metadata, constructs prompts, encodes with
    BioClinical ModernBERT, and projects to the shared embedding space.

    Args:
        model_name:   HuggingFace model ID for BioClinical ModernBERT.
        output_dim:   dimension of the shared embedding space (must match image encoder).
        hidden_dim:   hidden dimension in the projection head.
        dropout:      dropout rate in the projection head.
        freeze_bert:  if True, freeze all BERT parameters (default: True).
        pooling:      how to pool BERT outputs -- "cls" or "mean".
        max_length:   max token length for the tokenizer.
        templates:    custom templates dict (or None for defaults).

    Example::

        >>> encoder = MorphoCLIPTextEncoder()
        >>> metadata = [
        ...     {"modality": "compound", "compound_name": "PFI-1",
        ...      "smiles": "CC(=O)Nc1ccc(F)cc1", "target_gene": "BRD4",
        ...      "gene_function": "Bromodomain-containing transcription regulator",
        ...      "cell_line": "U2OS"},
        ... ]
        >>> embeddings = encoder(metadata)  # [1, 512]
    """

    def __init__(
        self,
        model_name: str = "thomas-sounack/BioClinical-ModernBERT-base",
        output_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_bert: bool = True,
        pooling: str = "cls",
        max_length: int = 256,
        templates: dict[str, str] | None = None,
    ) -> None:
        super().__init__()

        self.templates = templates or TEMPLATES

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size  # 768 for base

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

        self.freeze_bert = freeze_bert
        self.pooling = pooling
        self.max_length = max_length

        self.projection = ProjectionHead(
            input_dim=self.bert_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of text strings."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool BERT outputs to a single vector per input.

        Args:
            last_hidden_state: ``[B, seq_len, hidden_dim]``
            attention_mask:    ``[B, seq_len]``

        Returns:
            Pooled tensor of shape ``[B, hidden_dim]``.
        """
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]  # [CLS] token

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            summed = (last_hidden_state * mask).sum(dim=1)  # [B, hidden_dim]
            counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            return summed / counts

        raise ValueError(f"Unknown pooling method: {self.pooling}")

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode raw text strings (already constructed prompts) -> embeddings.

        Returns:
            Embeddings of shape ``[B, output_dim]``, L2 normalized.
        """
        device = next(self.projection.parameters()).device

        tokens = self.tokenize(texts)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # ModernBERT does NOT use token_type_ids
        tokens.pop("token_type_ids", None)

        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(**tokens)
        else:
            outputs = self.bert(**tokens)

        pooled = self.pool(outputs.last_hidden_state, tokens["attention_mask"])
        return self.projection(pooled)

    @torch.no_grad()
    def encode_texts_raw(self, texts: list[str]) -> torch.Tensor:
        """Encode texts through BERT only -- NO projection head.

        Returns raw 768-d [CLS] (or mean-pooled) features.  Use this for
        caching: cached raw features are deterministic and reusable
        regardless of how the projection head is trained.
        """
        device = next(self.bert.parameters()).device
        tokens = self.tokenize(texts)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        tokens.pop("token_type_ids", None)
        outputs = self.bert(**tokens)
        return self.pool(outputs.last_hidden_state, tokens["attention_mask"])

    def forward(self, metadata: list[dict]) -> torch.Tensor:
        """Full pipeline: metadata dicts -> prompts -> BERT -> projection -> embeddings.

        Args:
            metadata: list of dicts, each containing perturbation metadata.

        Returns:
            Embeddings of shape ``[B, output_dim]``, L2 normalized.
        """
        prompts = build_prompts(metadata, self.templates)
        return self.encode_texts(prompts)
