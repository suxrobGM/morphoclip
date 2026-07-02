"""Tests for MorphoCLIP loss functions."""

import pytest
import torch
import torch.nn.functional as F

from morphoclip.training.losses import (
    build_soft_labels,
    compute_loss,
    cwcl_loss,
    infonce_loss,
)


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
