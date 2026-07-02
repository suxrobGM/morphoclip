"""Record scoring, ranking, and leaderboard rows for the CellCLIP scheduler."""

from __future__ import annotations

from typing import Any


def _score_tuple(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    summary = record.get("analysis_summary", {})
    primary = summary.get("primary", summary)
    compound = primary.get("compound_eval_retrieval", {})
    overall = primary.get("eval_retrieval", {})
    metrics = primary.get("final_metrics", {})
    split_pca = primary.get("split_pca", {}).get("compound", {})
    image_pca = split_pca.get("image", {})
    text_pca = split_pca.get("text", {})
    return (
        -float(compound.get("broad_sample_R@1", 0.0)),
        -float(overall.get("broad_sample_R@1", 0.0)),
        -float(metrics.get("text_to_image_R@10", 0.0)),
        float(image_pca.get("top1_fraction", 1.0)),
        float(text_pca.get("top1_fraction", 1.0)),
    )


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort completed records using the fixed promotion order."""
    completed = [record for record in records if record.get("status") == "completed"]
    return sorted(completed, key=_score_tuple)


def _leaderboard_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, record in enumerate(rank_records(records), start=1):
        primary = record["analysis_summary"]["primary"]
        rows.append(
            {
                "rank": rank,
                "candidate_id": record["candidate_id"],
                "family_id": record["family_id"],
                "stage": record["stage"],
                "status": record["status"],
                "compound_broad_sample_R@1": primary["compound_eval_retrieval"].get(
                    "broad_sample_R@1", 0.0
                ),
                "overall_broad_sample_R@1": primary["eval_retrieval"].get("broad_sample_R@1", 0.0),
                "text_to_image_R@10": primary["final_metrics"].get("text_to_image_R@10", 0.0),
                "compound_image_top1_fraction": primary["split_pca"]["compound"]["image"].get(
                    "top1_fraction", 1.0
                ),
                "compound_text_top1_fraction": primary["split_pca"]["compound"]["text"].get(
                    "top1_fraction", 1.0
                ),
                "run_dir": record["run_dir"],
                "analysis_summary_path": record["analysis_summary_path"],
            }
        )
    return rows
