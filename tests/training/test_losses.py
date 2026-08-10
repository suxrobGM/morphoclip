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
SCALE = torch.tensor(2.6593)  # log(1 / 0.07)


def _pair(n: int = 4, dim: int = 16, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic pair of L2-normalized embedding batches, so failures reproduce."""
    generator = torch.Generator().manual_seed(seed)
    return (
        F.normalize(torch.randn(n, dim, generator=generator), dim=-1),
        F.normalize(torch.randn(n, dim, generator=generator), dim=-1),
    )


def build_soft_labels(broad_samples, **kwargs):
    """Row-normalized labels, the form ``cwcl_loss`` feeds to cross-entropy."""
    return _row_normalize(build_affinity_matrix(broad_samples, **kwargs))


class TestInfoNCELoss:
    def test_perfect_alignment_gives_near_zero_loss(self) -> None:
        emb, _ = _pair()
        assert infonce_loss(emb, emb, SCALE).item() < 0.1

    def test_symmetry(self) -> None:
        image, text = _pair(8, 32)
        torch.testing.assert_close(
            infonce_loss(image, text, SCALE), infonce_loss(text, image, SCALE), atol=1e-5, rtol=1e-5
        )

    def test_gradient_flows_to_the_embeddings_and_the_temperature(self) -> None:
        raw = torch.randn(4, 16, requires_grad=True)
        _, text = _pair()
        scale = torch.tensor(2.6593, requires_grad=True)
        infonce_loss(F.normalize(raw, dim=-1), text, scale).backward()
        assert raw.grad is not None
        assert scale.grad is not None


class TestAffinityMatrix:
    def test_replicates_share_the_row_evenly(self) -> None:
        labels = build_soft_labels(["X", "X", "Y"], device=CPU)
        assert labels.shape == (3, 3)
        torch.testing.assert_close(labels[0], torch.tensor([0.5, 0.5, 0.0]))
        torch.testing.assert_close(labels[2, 2], torch.tensor(1.0))

    def test_zero_weight_matches_binary_labels(self) -> None:
        """target_weight=0 must reproduce the binary same-broad_sample labels."""
        broad = ["A", "A", "B", "C"]
        keys = ["GENE1", "GENE1", "GENE1", "GENE2"]
        torch.testing.assert_close(
            build_soft_labels(broad, target_keys=keys, target_weight=0.0, device=CPU),
            build_soft_labels(broad, device=CPU),
        )

    def test_shared_gene_gets_target_weight(self) -> None:
        """A compound and a CRISPR knockout sharing a gene get target_weight."""
        affinity = build_affinity_matrix(
            ["CMPD", "KO"], target_keys=["GENE1|GENE2", "GENE1"], target_weight=0.6, device=CPU
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(0.6))
        torch.testing.assert_close(affinity[1, 0], torch.tensor(0.6))
        torch.testing.assert_close(affinity[0, 0], torch.tensor(1.0))

    @pytest.mark.parametrize("target_keys", [["GENE1", "GENE2"], ["", ""], ["", "GENE1"]])
    def test_targets_that_do_not_overlap_get_zero(self, target_keys: list[str]) -> None:
        """An unknown target is not a shared target."""
        affinity = build_affinity_matrix(
            ["A", "B"], target_keys=target_keys, target_weight=0.7, device=CPU
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(0.0))
        torch.testing.assert_close(affinity[1, 0], torch.tensor(0.0))

    def test_same_broad_sample_keeps_full_weight(self) -> None:
        """Replicates stay at 1.0 even when target_weight is lower."""
        affinity = build_affinity_matrix(
            ["A", "A"], target_keys=["GENE1", "GENE1"], target_weight=0.5, device=CPU
        )
        torch.testing.assert_close(affinity[0, 1], torch.tensor(1.0))

    def test_both_directions_row_sum_to_one(self) -> None:
        """Asymmetric group sizes plus continuous weights: both directions normalize."""
        broad = ["A", "A", "A", "B", "C"]
        keys = ["GENE1", "GENE1", "GENE1", "GENE1", "GENE2"]
        affinity = build_affinity_matrix(broad, target_keys=keys, target_weight=0.6, device=CPU)
        i2t = affinity / affinity.sum(dim=1, keepdim=True)
        t2i = affinity.t() / affinity.t().sum(dim=1, keepdim=True)
        torch.testing.assert_close(i2t.sum(dim=1), torch.ones(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(t2i.sum(dim=1), torch.ones(5), atol=1e-6, rtol=0)
        # The naive transpose of the row-normalized matrix does NOT sum to 1.
        assert not torch.allclose(i2t.t().sum(dim=1), torch.ones(5), atol=1e-3)


class TestCWCLLoss:
    @pytest.mark.parametrize(
        "target_kwargs", [{}, {"target_keys": ["G1", "G2", "G3", "G4"], "target_weight": 0.6}]
    )
    def test_unique_samples_match_infonce(self, target_kwargs: dict) -> None:
        """No soft positives, with or without gene weights, means plain InfoNCE."""
        image, text = _pair()
        torch.testing.assert_close(
            cwcl_loss(image, text, SCALE, broad_samples=["A", "B", "C", "D"], **target_kwargs),
            infonce_loss(image, text, SCALE),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_target_weight_changes_loss(self) -> None:
        image, text = _pair()
        broad = ["A", "B", "C", "D"]
        binary = cwcl_loss(image, text, SCALE, broad_samples=broad)
        gene_aware = cwcl_loss(
            image,
            text,
            SCALE,
            broad_samples=broad,
            target_keys=["G1", "G1", "G2", "G2"],
            target_weight=0.6,
        )
        assert not torch.allclose(binary, gene_aware, atol=1e-4)


class TestReplicateImageLoss:
    def test_no_positives_returns_zero_with_grad(self) -> None:
        raw = torch.randn(4, 16, requires_grad=True)
        loss = replicate_image_loss(
            F.normalize(raw, dim=-1), 10.0, broad_samples=["A", "B", "C", "D"]
        )
        assert loss.item() == 0.0
        loss.backward()
        assert raw.grad is not None

    def test_pulls_replicates_together(self) -> None:
        """Closer replicate embeddings give a lower loss."""
        broad = ["A", "A", "B", "B"]
        base_a, base_b = _pair(1, 16)
        noise, _ = _pair(2, 16, seed=1)

        far = F.normalize(torch.cat([base_a, noise[:1], base_b, noise[1:]], dim=0), dim=-1)
        close = F.normalize(torch.cat([base_a, base_a, base_b, base_b], dim=0), dim=-1)
        assert (
            replicate_image_loss(close, 10.0, broad_samples=broad).item()
            < replicate_image_loss(far, 10.0, broad_samples=broad).item()
        )

    def test_diagonal_excluded(self) -> None:
        """Identical replicates give a near-zero loss only if self is excluded."""
        emb, _ = _pair(1, 16)
        loss = replicate_image_loss(emb.repeat(2, 1), 10.0, broad_samples=["A", "A"])
        # Excluding self leaves one candidate, so log(1) = 0.
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)


class TestComputeLoss:
    def test_dispatch_matches_the_underlying_loss(self) -> None:
        image, text = _pair()
        broad = ["A", "A", "B", "B"]
        torch.testing.assert_close(
            compute_loss("infonce", image, text, SCALE), infonce_loss(image, text, SCALE)
        )
        torch.testing.assert_close(
            compute_loss("cwcl", image, text, SCALE, broad_samples=broad),
            cwcl_loss(image, text, SCALE, broad_samples=broad),
        )

    @pytest.mark.parametrize(
        ("loss_type", "message"), [("cwcl", "broad_samples"), ("unknown", "Unknown loss_type")]
    )
    def test_bad_arguments_raise(self, loss_type: str, message: str) -> None:
        image, text = _pair()
        with pytest.raises(ValueError, match=message):
            compute_loss(loss_type, image, text, SCALE)


class TestComputeTrainingLoss:
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        image, text = _pair(6, 16, seed=11)
        return image, text, ["A", "A", "B", "B", "C", "C"]

    def test_no_components_when_replicate_disabled(self) -> None:
        image, text, broad = self._inputs()
        total, components = compute_training_loss("cwcl", image, text, SCALE, broad_samples=broad)
        torch.testing.assert_close(
            total, compute_loss("cwcl", image, text, SCALE, broad_samples=broad)
        )
        assert components == {}

    def test_total_is_text_plus_weighted_replicate(self) -> None:
        image, text, broad = self._inputs()
        total, components = compute_training_loss(
            "cwcl", image, text, SCALE, broad_samples=broad, replicate_weight=0.3
        )
        text_loss = compute_loss("cwcl", image, text, SCALE, broad_samples=broad)
        replicate = replicate_image_loss(image, SCALE.exp(), broad_samples=broad)

        torch.testing.assert_close(total, text_loss + 0.3 * replicate)
        assert set(components) == {"text", "replicate"}
        torch.testing.assert_close(components["text"], text_loss.detach())
        torch.testing.assert_close(components["replicate"], replicate.detach())

    def test_components_are_detached(self) -> None:
        image, text, broad = self._inputs()
        total, components = compute_training_loss(
            "cwcl",
            image.requires_grad_(True),
            text,
            SCALE,
            broad_samples=broad,
            replicate_weight=0.3,
        )
        assert total.requires_grad
        assert not any(value.requires_grad for value in components.values())

    def test_fixed_temperature_overrides_learnable_scale(self) -> None:
        image, text, broad = self._inputs()
        _, components = compute_training_loss(
            "cwcl",
            image,
            text,
            SCALE,
            broad_samples=broad,
            replicate_weight=1.0,
            replicate_temperature=0.1,
        )
        expected = replicate_image_loss(image, 1.0 / 0.1, broad_samples=broad)
        torch.testing.assert_close(components["replicate"], expected.detach())

    def test_replicate_without_broad_samples_raises(self) -> None:
        image, text, _ = self._inputs()
        with pytest.raises(ValueError, match="broad_samples"):
            compute_training_loss("infonce", image, text, SCALE, replicate_weight=0.3)
