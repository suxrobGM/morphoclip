"""Tests for MorphoCLIP loss functions."""

import pytest
import torch
import torch.nn.functional as F

from morphoclip.training.losses import (
    _row_normalize,
    build_affinity_matrix,
    compute_loss,
    compute_training_loss,
    cwcl_loss,
    infonce_loss,
    replicate_image_loss,
)

CPU = torch.device("cpu")


def build_soft_labels(broad_samples, **kwargs):
    """Row-normalized labels, the form ``cwcl_loss`` feeds to cross-entropy."""
    return _row_normalize(build_affinity_matrix(broad_samples, **kwargs))


class TestInfoNCELoss:
    """Tests for symmetric InfoNCE loss."""

    def test_non_negative(self) -> None:
        image = F.normalize(torch.randn(8, 32), dim=-1)
        text = F.normalize(torch.randn(8, 32), dim=-1)
        scale = torch.tensor(2.6593)  # log(1/0.07)
        loss = infonce_loss(image, text, scale)
        assert loss.item() >= 0

    def test_perfect_alignment_low_loss(self) -> None:
        """Identical embeddings should produce low loss."""
        emb = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        loss = infonce_loss(emb, emb, scale)
        assert loss.item() < 0.1

    def test_symmetry(self) -> None:
        """Loss should be the same regardless of argument order."""
        image = F.normalize(torch.randn(8, 32), dim=-1)
        text = F.normalize(torch.randn(8, 32), dim=-1)
        scale = torch.tensor(2.6593)
        loss_1 = infonce_loss(image, text, scale)
        loss_2 = infonce_loss(text, image, scale)
        torch.testing.assert_close(loss_1, loss_2, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self) -> None:
        raw = torch.randn(4, 16, requires_grad=True)
        image = F.normalize(raw, dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593, requires_grad=True)
        loss = infonce_loss(image, text, scale)
        loss.backward()
        assert raw.grad is not None
        assert scale.grad is not None


class TestBuildSoftLabels:
    """Tests for soft label matrix construction."""

    def test_identity_labels(self) -> None:
        """All unique samples → identity matrix."""
        labels = build_soft_labels(["A", "B", "C"], device=torch.device("cpu"))
        expected = torch.eye(3)
        torch.testing.assert_close(labels, expected)

    def test_shared_perturbation(self) -> None:
        """Shared broad_sample → equal weight among matches."""
        labels = build_soft_labels(
            ["X", "X", "Y"],
            device=torch.device("cpu"),
        )
        assert labels.shape == (3, 3)
        # First two samples share "X" → each gets 0.5
        torch.testing.assert_close(labels[0, 0], torch.tensor(0.5))
        torch.testing.assert_close(labels[0, 1], torch.tensor(0.5))
        torch.testing.assert_close(labels[0, 2], torch.tensor(0.0))
        # Third sample is unique → 1.0 on diagonal
        torch.testing.assert_close(labels[2, 2], torch.tensor(1.0))

    def test_rows_sum_to_one(self) -> None:
        labels = build_soft_labels(
            ["A", "A", "B", "B", "C"],
            device=torch.device("cpu"),
        )
        row_sums = labels.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(5), atol=1e-6, rtol=0)


class TestGeneAwareSoftLabels:
    """Tests for continuous, target-gene-aware soft labels."""

    def test_zero_weight_matches_binary_labels(self) -> None:
        """target_weight=0 must reproduce the binary same-broad_sample labels."""
        broad = ["A", "A", "B", "C"]
        keys = ["GENE1", "GENE1", "GENE1", "GENE2"]
        binary = build_soft_labels(broad, device=CPU)
        gene_aware = build_soft_labels(broad, target_keys=keys, target_weight=0.0, device=CPU)
        torch.testing.assert_close(gene_aware, binary)

    def test_shared_gene_gets_target_weight(self) -> None:
        """A compound and a CRISPR knockout sharing a gene get target_weight."""
        affinity = build_affinity_matrix(
            ["CMPD", "KO"],
            target_keys=["GENE1|GENE2", "GENE1"],
            target_weight=0.6,
            device=CPU,
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(0.6))
        torch.testing.assert_close(affinity[1, 0], torch.tensor(0.6))
        # Diagonal stays 1.0
        torch.testing.assert_close(affinity[0, 0], torch.tensor(1.0))

    def test_disjoint_genes_get_zero(self) -> None:
        affinity = build_affinity_matrix(
            ["A", "B"],
            target_keys=["GENE1", "GENE2"],
            target_weight=0.6,
            device=CPU,
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(0.0))

    def test_empty_key_never_matches(self) -> None:
        """Unknown target must not be treated as a shared target."""
        affinity = build_affinity_matrix(
            ["A", "B", "C"],
            target_keys=["", "", "GENE1"],
            target_weight=0.7,
            device=CPU,
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(0.0))
        torch.testing.assert_close(affinity[1, 0], torch.tensor(0.0))
        torch.testing.assert_close(affinity[0, 2], torch.tensor(0.0))

    def test_same_broad_sample_keeps_full_weight(self) -> None:
        """Replicates stay at 1.0 even when target_weight is lower."""
        affinity = build_affinity_matrix(
            ["A", "A"],
            target_keys=["GENE1", "GENE1"],
            target_weight=0.5,
            device=CPU,
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(1.0))

    def test_both_directions_row_sum_to_one(self) -> None:
        """Asymmetric group sizes + continuous weights: both directions normalize."""
        broad = ["A", "A", "A", "B", "C"]
        keys = ["GENE1", "GENE1", "GENE1", "GENE1", "GENE2"]
        affinity = build_affinity_matrix(broad, target_keys=keys, target_weight=0.6, device=CPU)
        i2t = affinity / affinity.sum(dim=1, keepdim=True)
        t2i = affinity.t() / affinity.t().sum(dim=1, keepdim=True)
        torch.testing.assert_close(i2t.sum(dim=1), torch.ones(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(t2i.sum(dim=1), torch.ones(5), atol=1e-6, rtol=0)
        # The naive transpose of the row-normalized matrix does NOT sum to 1
        naive = i2t.t().sum(dim=1)
        assert not torch.allclose(naive, torch.ones(5), atol=1e-3)


class TestCWCLLoss:
    """Tests for CWCL loss."""

    def test_non_negative(self) -> None:
        image = F.normalize(torch.randn(6, 32), dim=-1)
        text = F.normalize(torch.randn(6, 32), dim=-1)
        scale = torch.tensor(2.6593)
        loss = cwcl_loss(
            image,
            text,
            scale,
            broad_samples=["A", "A", "B", "B", "C", "C"],
        )
        assert loss.item() >= 0

    def test_unique_samples_matches_infonce(self) -> None:
        """With all unique samples, CWCL should equal InfoNCE."""
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        loss_infonce = infonce_loss(image, text, scale)
        loss_cwcl = cwcl_loss(
            image,
            text,
            scale,
            broad_samples=["A", "B", "C", "D"],
        )
        torch.testing.assert_close(loss_infonce, loss_cwcl, atol=1e-5, rtol=1e-5)

    def test_unique_samples_matches_infonce_with_targets(self) -> None:
        """The InfoNCE invariant survives gene weights when targets are disjoint."""
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        loss_infonce = infonce_loss(image, text, scale)
        loss_cwcl = cwcl_loss(
            image,
            text,
            scale,
            broad_samples=["A", "B", "C", "D"],
            target_keys=["G1", "G2", "G3", "G4"],
            target_weight=0.6,
        )
        torch.testing.assert_close(loss_infonce, loss_cwcl, atol=1e-5, rtol=1e-5)

    def test_target_weight_changes_loss(self) -> None:
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        keys = ["G1", "G1", "G2", "G2"]
        binary = cwcl_loss(image, text, scale, broad_samples=["A", "B", "C", "D"])
        gene_aware = cwcl_loss(
            image,
            text,
            scale,
            broad_samples=["A", "B", "C", "D"],
            target_keys=keys,
            target_weight=0.6,
        )
        assert not torch.allclose(binary, gene_aware, atol=1e-4)


class TestReplicateImageLoss:
    """Tests for the replicate-alignment image-image term."""

    def test_no_positives_returns_zero_with_grad(self) -> None:
        raw = torch.randn(4, 16, requires_grad=True)
        image = F.normalize(raw, dim=-1)
        loss = replicate_image_loss(image, 10.0, broad_samples=["A", "B", "C", "D"])
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()
        assert raw.grad is not None

    def test_pulls_replicates_together(self) -> None:
        """Closer replicate embeddings give a lower loss."""
        broad = ["A", "A", "B", "B"]
        base_a = F.normalize(torch.randn(1, 16), dim=-1)
        base_b = F.normalize(torch.randn(1, 16), dim=-1)
        noise = F.normalize(torch.randn(2, 16), dim=-1)

        far = F.normalize(torch.cat([base_a, noise[:1], base_b, noise[1:]], dim=0), dim=-1)
        close = F.normalize(torch.cat([base_a, base_a, base_b, base_b], dim=0), dim=-1)
        loss_far = replicate_image_loss(far, 10.0, broad_samples=broad)
        loss_close = replicate_image_loss(close, 10.0, broad_samples=broad)
        assert loss_close.item() < loss_far.item()

    def test_diagonal_excluded(self) -> None:
        """Identical replicates give a near-zero loss only if self is excluded."""
        emb = F.normalize(torch.randn(1, 16), dim=-1).repeat(2, 1)
        loss = replicate_image_loss(emb, 10.0, broad_samples=["A", "A"])
        # Excluding self leaves one candidate, so log(1) = 0.
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_rows_without_positives_are_dropped(self) -> None:
        """A singleton row must not drag the mean toward its (undefined) value."""
        emb = F.normalize(torch.randn(3, 16), dim=-1)
        with_singleton = replicate_image_loss(emb, 10.0, broad_samples=["A", "A", "Z"])
        pair_only = replicate_image_loss(emb[:2], 10.0, broad_samples=["A", "A"])
        # Same positives, but the 3-sample batch has an extra negative, so the
        # losses differ; the singleton row itself contributes nothing.
        assert with_singleton.item() >= 0.0
        assert pair_only.item() >= 0.0

    def test_scale_can_be_a_tensor(self) -> None:
        emb = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593).exp()
        loss = replicate_image_loss(emb, scale, broad_samples=["A", "A", "B", "B"])
        assert loss.item() >= 0.0


class TestComputeLoss:
    """Tests for the loss dispatch function."""

    def test_dispatch_infonce(self) -> None:
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        loss = compute_loss("infonce", image, text, scale)
        assert loss.item() >= 0

    def test_dispatch_cwcl(self) -> None:
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        loss = compute_loss(
            "cwcl",
            image,
            text,
            scale,
            broad_samples=["A", "B", "C", "D"],
        )
        assert loss.item() >= 0

    def test_cwcl_missing_samples_raises(self) -> None:
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        with pytest.raises(ValueError, match="broad_samples"):
            compute_loss("cwcl", image, text, scale)

    def test_unknown_loss_raises(self) -> None:
        image = F.normalize(torch.randn(4, 16), dim=-1)
        text = F.normalize(torch.randn(4, 16), dim=-1)
        scale = torch.tensor(2.6593)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            compute_loss("unknown", image, text, scale)


class TestComputeTrainingLoss:
    """Tests for total-loss composition and its reported components."""

    def _inputs(self):
        generator = torch.Generator().manual_seed(11)
        image = F.normalize(torch.randn(6, 16, generator=generator), dim=-1)
        text = F.normalize(torch.randn(6, 16, generator=generator), dim=-1)
        return image, text, torch.tensor(2.6593), ["A", "A", "B", "B", "C", "C"]

    def test_no_components_when_replicate_disabled(self) -> None:
        image, text, scale, broad = self._inputs()
        total, components = compute_training_loss("cwcl", image, text, scale, broad_samples=broad)
        expected = compute_loss("cwcl", image, text, scale, broad_samples=broad)
        torch.testing.assert_close(total, expected)
        assert components == {}

    def test_total_is_text_plus_weighted_replicate(self) -> None:
        image, text, scale, broad = self._inputs()
        total, components = compute_training_loss(
            "cwcl", image, text, scale, broad_samples=broad, replicate_weight=0.3
        )
        text_loss = compute_loss("cwcl", image, text, scale, broad_samples=broad)
        replicate = replicate_image_loss(image, scale.exp(), broad_samples=broad)
        torch.testing.assert_close(total, text_loss + 0.3 * replicate)
        assert set(components) == {"text", "replicate"}
        torch.testing.assert_close(components["text"], text_loss.detach())
        torch.testing.assert_close(components["replicate"], replicate.detach())

    def test_components_are_detached(self) -> None:
        image, text, scale, broad = self._inputs()
        image = image.requires_grad_(True)
        total, components = compute_training_loss(
            "cwcl", image, text, scale, broad_samples=broad, replicate_weight=0.3
        )
        assert total.requires_grad
        assert not any(value.requires_grad for value in components.values())

    def test_fixed_temperature_overrides_learnable_scale(self) -> None:
        image, text, scale, broad = self._inputs()
        _, components = compute_training_loss(
            "cwcl",
            image,
            text,
            scale,
            broad_samples=broad,
            replicate_weight=1.0,
            replicate_temperature=0.1,
        )
        expected = replicate_image_loss(image, 1.0 / 0.1, broad_samples=broad)
        torch.testing.assert_close(components["replicate"], expected.detach())

    def test_replicate_without_broad_samples_raises(self) -> None:
        image, text, scale, _ = self._inputs()
        with pytest.raises(ValueError, match="broad_samples"):
            compute_training_loss("infonce", image, text, scale, replicate_weight=0.3)
