"""Serialization, comparison, and Markdown reporting for CellCLIP run analysis."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def to_serializable(value: Any) -> Any:
    """Recursively convert analysis values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def load_benchmark_tables(benchmark_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load benchmark summary tables when available."""
    tables_dir = benchmark_dir / "tables"
    if not tables_dir.exists():
        return {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for csv_path in sorted(tables_dir.glob("*summary.csv")):
        frame = pd.read_csv(csv_path)
        tables[csv_path.stem] = frame.to_dict(orient="records")
    return tables


def build_comparison(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Build a compact comparison between two run summaries."""
    comparison: dict[str, Any] = {}
    for key in (
        "eval_retrieval",
        "compound_eval_retrieval",
        "image_pca",
        "text_pca",
        "fusion_diagnostics",
    ):
        comparison[key] = {}
        for metric_name, primary_value in primary.get(key, {}).items():
            secondary_value = secondary.get(key, {}).get(metric_name)
            if secondary_value is None:
                continue
            comparison[key][metric_name] = {
                "primary": primary_value,
                "secondary": secondary_value,
                "delta": primary_value - secondary_value,
            }
    return comparison


def render_report(
    primary: dict[str, Any],
    *,
    secondary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable Markdown report."""
    lines = [
        "# CellCLIP Run Analysis",
        "",
        f"- Run: `{primary['run_dir']}`",
        f"- Checkpoint: `{primary['checkpoint_path']}`",
        "",
        "## Eval Retrieval",
    ]
    for key, value in primary["eval_retrieval"].items():
        lines.append(f"- {key}: {value:.6f}")
    duplicate_train = json.dumps(
        to_serializable(primary["duplicate_stats"]["train"]), sort_keys=True
    )
    duplicate_eval = json.dumps(to_serializable(primary["duplicate_stats"]["eval"]), sort_keys=True)
    fusion_diagnostics = json.dumps(to_serializable(primary["fusion_diagnostics"]), sort_keys=True)
    lines.extend(
        [
            "",
            "## Compound Retrieval",
            *(f"- {key}: {value:.6f}" for key, value in primary["compound_eval_retrieval"].items()),
            "",
            "## Perturbation Retrieval",
            *(
                f"- {name}: {json.dumps(to_serializable(metrics), sort_keys=True)}"
                for name, metrics in primary["perturbation_retrieval"].items()
            ),
            "",
            "## Duplicate Stats",
            f"- Train: {duplicate_train}",
            f"- Eval: {duplicate_eval}",
            "",
            "## Chem Diagnostics",
            f"- Has SMILES fraction: {primary['has_smiles_fraction']:.6f}",
            f"- Fusion: {fusion_diagnostics}",
            "",
            "## PCA Diagnostics",
            f"- Image: {json.dumps(to_serializable(primary['image_pca']), sort_keys=True)}",
            f"- Text: {json.dumps(to_serializable(primary['text_pca']), sort_keys=True)}",
            f"- Split: {json.dumps(to_serializable(primary['split_pca']), sort_keys=True)}",
        ]
    )
    if primary.get("benchmark_tables"):
        lines.extend(["", "## Benchmark Tables"])
        for name, rows in primary["benchmark_tables"].items():
            lines.append(f"- {name}: {len(rows)} rows")
    if secondary is not None and comparison is not None:
        lines.extend(["", "## Comparison", f"- Compare run: `{secondary['run_dir']}`"])
        for section, metrics in comparison.items():
            lines.append(f"- {section}: {json.dumps(to_serializable(metrics), sort_keys=True)}")
    return "\n".join(lines) + "\n"


def write_analysis_outputs(
    output_dir: Path,
    primary: dict[str, Any],
    *,
    secondary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write analysis outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    payload = {"primary": primary}
    if secondary is not None:
        payload["secondary"] = secondary
    if comparison is not None:
        payload["comparison"] = comparison
    summary_path.write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        render_report(primary, secondary=secondary, comparison=comparison),
        encoding="utf-8",
    )
    return summary_path, report_path
