"""Tests for corrected retrieval metrics (`morphoclip.training.retrieval`)."""

import itertools
import math

import torch

from morphoclip.training.retrieval import (
    aggregate_images,
    compute_retrieval_metrics,
    dedupe_texts,
)


def _basis(dim: int, index: int) -> torch.Tensor:
    vec = torch.zeros(dim)
    vec[index] = 1.0
    return vec


class TestClumpingRegression:
    """Replicate wells must not inflate the candidate pool."""

    @staticmethod
    def _fixture() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        # 3 perturbations x 4 replicates; identical text vector per perturbation.
        texts = [_basis(3, 0), _basis(3, 1), _basis(3, 2)]
        # Each well scores its OWN text second-highest of the 3 unique texts.
        images = [
            0.5 * _basis(3, 0) + 0.6 * _basis(3, 1) + 0.1 * _basis(3, 2),
            0.1 * _basis(3, 0) + 0.5 * _basis(3, 1) + 0.6 * _basis(3, 2),
            0.6 * _basis(3, 0) + 0.1 * _basis(3, 1) + 0.5 * _basis(3, 2),
        ]
        image_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        broad_samples: list[str] = []
        for pert in range(3):
            for _ in range(4):
                image_rows.append(images[pert])
                text_rows.append(texts[pert])
                broad_samples.append(f"pert-{pert}")
        return torch.stack(image_rows), torch.stack(text_rows), broad_samples

    def test_pool_is_unique_perturbations(self) -> None:
        images, texts, samples = self._fixture()
        metrics = compute_retrieval_metrics(images, texts, broad_samples=samples)

        assert metrics["n_wells"] == 12.0
        assert metrics["n_perturbations"] == 3.0
        # Every well's true text ranks 2nd of 3: never top-1, always top-5.
        assert metrics["image_to_text_R@1"] == 0.0
        assert metrics["image_to_text_R@5"] == 1.0
        assert metrics["image_to_text_mean_rank"] == 2.0
        assert metrics["image_to_text_median_rank"] == 2.0

    def test_pert_level_keys_present(self) -> None:
        images, texts, samples = self._fixture()
        metrics = compute_retrieval_metrics(images, texts, broad_samples=samples)
        for key in (
            "pert_image_to_text_R@1",
            "pert_image_to_text_mean_rank",
            "pert_image_to_text_median_rank",
            "pert_text_to_image_R@10",
            "pert_text_to_image_mean_rank",
        ):
            assert key in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestPerfectAlignment:
    """When image == text for every well, all directions score R@1 == 1."""

    def test_all_directions(self) -> None:
        n_pert, n_rep, dim = 4, 3, 4
        rows: list[torch.Tensor] = []
        samples: list[str] = []
        for pert in range(n_pert):
            for _ in range(n_rep):
                rows.append(_basis(dim, pert))
                samples.append(f"p{pert}")
        emb = torch.stack(rows)

        metrics = compute_retrieval_metrics(emb, emb.clone(), broad_samples=samples)
        assert metrics["image_to_text_R@1"] == 1.0
        assert metrics["text_to_image_R@1"] == 1.0
        assert metrics["pert_image_to_text_R@1"] == 1.0
        assert metrics["pert_text_to_image_R@1"] == 1.0
        assert metrics["image_to_text_mean_rank"] == 1.0
        assert metrics["text_to_image_mean_rank"] == 1.0


class TestRandomBaselines:
    def test_single_positive_baseline_is_k_over_p(self) -> None:
        n_pert, n_rep, dim = 12, 2, 8
        generator = torch.Generator().manual_seed(0)
        samples = [f"p{p}" for p in range(n_pert) for _ in range(n_rep)]
        images = torch.randn(n_pert * n_rep, dim, generator=generator)
        texts = torch.stack([_basis(dim, p % dim) for p in range(n_pert) for _ in range(n_rep)])

        metrics = compute_retrieval_metrics(images, texts, broad_samples=samples)
        for key_prefix in ("image_to_text", "pert_image_to_text", "pert_text_to_image"):
            for k in (1, 5, 10):
                assert metrics[f"{key_prefix}_random_R@{k}"] == k / n_pert
            assert metrics[f"{key_prefix}_random_mean_rank"] == (n_pert + 1) / 2
            assert metrics[f"{key_prefix}_random_median_rank"] == (n_pert + 1) / 2

    def test_single_positive_baseline_clamps_when_k_exceeds_pool(self) -> None:
        samples = [f"p{p}" for p in range(3) for _ in range(2)]
        emb = torch.randn(6, 4, generator=torch.Generator().manual_seed(1))
        metrics = compute_retrieval_metrics(emb, emb.clone(), broad_samples=samples)
        assert metrics["image_to_text_random_R@1"] == 1 / 3
        assert metrics["image_to_text_random_R@5"] == 1.0
        assert metrics["image_to_text_random_R@10"] == 1.0

    def test_any_of_m_baseline_matches_brute_force(self) -> None:
        # N = 6 wells, P = 2 perturbations, m = 3 replicates each.
        n_wells, n_rep = 6, 3
        samples = [f"p{p}" for p in range(2) for _ in range(n_rep)]
        emb = torch.randn(n_wells, 5, generator=torch.Generator().manual_seed(2))
        metrics = compute_retrieval_metrics(emb, emb.clone(), broad_samples=samples)

        positives = {0, 1, 2}
        for k in (1, 5):
            subsets = list(itertools.combinations(range(n_wells), k))
            hit = sum(1 for s in subsets if positives & set(s))
            assert math.isclose(
                metrics[f"text_to_image_random_R@{k}"], hit / len(subsets), rel_tol=1e-9
            )
        # k = 10 > N: every ordering contains a positive.
        assert metrics["text_to_image_random_R@10"] == 1.0

        # Expected rank of the first positive, by exhaustive permutation.
        perms = list(itertools.permutations(range(n_wells)))
        expected = sum(min(p.index(w) for w in positives) + 1 for p in perms) / len(perms)
        assert math.isclose(metrics["text_to_image_random_mean_rank"], expected, rel_tol=1e-9)
        assert math.isclose(
            metrics["text_to_image_random_mean_rank"], (n_wells + 1) / (n_rep + 1), rel_tol=1e-9
        )


class TestDedupeAndAggregate:
    def test_dedupe_first_occurrence_order(self) -> None:
        samples = ["b", "a", "b", "c", "a"]
        texts = torch.stack([_basis(3, i % 3) * float(i + 1) for i in range(5)])
        unique_texts, unique_ids, well_to_pert = dedupe_texts(texts, samples)

        assert unique_ids == ["b", "a", "c"]
        assert unique_texts.shape == (3, 3)
        assert torch.equal(well_to_pert, torch.tensor([0, 1, 0, 2, 1]))
        assert torch.equal(unique_texts[0], texts[0])
        assert torch.equal(unique_texts[1], texts[1])
        assert torch.equal(unique_texts[2], texts[3])

    def test_aggregate_images_unit_norm_and_order(self) -> None:
        samples = ["b", "a", "b", "c", "a"]
        images = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 4.0, 0.0],
            ]
        )
        profiles = aggregate_images(images, samples)

        assert profiles.shape == (3, 3)
        norms = profiles.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-6)
        # Row order follows first-occurrence order: b, a, c.
        assert torch.allclose(profiles[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
        assert torch.allclose(profiles[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
        assert torch.allclose(profiles[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)

    def test_aggregate_matches_dedupe_order(self) -> None:
        samples = ["z", "y", "z", "x"]
        emb = torch.randn(4, 6, generator=torch.Generator().manual_seed(3))
        _, unique_ids, _ = dedupe_texts(emb, samples)
        profiles = aggregate_images(emb, samples)
        assert unique_ids == ["z", "y", "x"]
        assert profiles.shape[0] == len(unique_ids)


class TestDiagonalFallback:
    def test_identity_embeddings_recall_one(self) -> None:
        emb = torch.eye(6)
        metrics = compute_retrieval_metrics(emb, emb.clone())

        assert metrics["image_to_text_R@1"] == 1.0
        assert metrics["text_to_image_R@1"] == 1.0
        assert metrics["image_to_text_mean_rank"] == 1.0
        assert metrics["image_to_text_median_rank"] == 1.0

    def test_no_random_or_pert_keys(self) -> None:
        emb = torch.eye(6)
        metrics = compute_retrieval_metrics(emb, emb.clone())
        assert not any("random" in key for key in metrics)
        assert not any(key.startswith("pert_") for key in metrics)
        assert "n_wells" not in metrics
        assert set(metrics) == {
            "image_to_text_R@1",
            "image_to_text_R@5",
            "image_to_text_R@10",
            "image_to_text_mean_rank",
            "image_to_text_median_rank",
            "text_to_image_R@1",
            "text_to_image_R@5",
            "text_to_image_R@10",
            "text_to_image_mean_rank",
            "text_to_image_median_rank",
        }


def test_defensive_normalization_is_scale_invariant() -> None:
    generator = torch.Generator().manual_seed(4)
    samples = [f"p{p}" for p in range(4) for _ in range(3)]
    images = torch.randn(12, 8, generator=generator)
    texts = torch.stack([_basis(8, p) for p in range(4) for _ in range(3)])

    baseline = compute_retrieval_metrics(
        torch.nn.functional.normalize(images, dim=-1),
        torch.nn.functional.normalize(texts, dim=-1),
        broad_samples=samples,
    )
    scaled_images = images.clone()
    scaled_images[0] *= 5.0
    scaled_texts = texts.clone()
    scaled_texts[7] *= 5.0
    scaled = compute_retrieval_metrics(scaled_images, scaled_texts, broad_samples=samples)

    assert baseline.keys() == scaled.keys()
    for key in baseline:
        assert math.isclose(baseline[key], scaled[key], rel_tol=1e-6, abs_tol=1e-6)


def test_backward_compatible_import_from_evaluate() -> None:
    from morphoclip.training.evaluate import compute_retrieval_metrics as legacy

    assert legacy is compute_retrieval_metrics
