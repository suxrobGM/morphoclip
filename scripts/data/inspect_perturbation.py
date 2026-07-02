"""Inspect the perturbation types and metadata model used by MorphoCLIP.

Examples:
    python scripts/inspect_perturbation.py
    python scripts/inspect_perturbation.py --plate BR00116991 --well A01
    python scripts/inspect_perturbation.py --example-type compound --json
"""

import importlib.util
import json
import sys
import types
from dataclasses import fields
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = SRC_ROOT / "morphoclip" / "data"


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_namespace_packages() -> None:
    morphoclip_pkg = sys.modules.setdefault("morphoclip", types.ModuleType("morphoclip"))
    if not hasattr(morphoclip_pkg, "__path__"):
        morphoclip_pkg.__path__ = [str(SRC_ROOT / "morphoclip")]

    data_pkg = sys.modules.setdefault("morphoclip.data", types.ModuleType("morphoclip.data"))
    if not hasattr(data_pkg, "__path__"):
        data_pkg.__path__ = [str(DATA_ROOT)]


_ensure_namespace_packages()
perturbation_module = _load_module("morphoclip.data.perturbation", DATA_ROOT / "perturbation.py")

PerturbationInfo = perturbation_module.PerturbationInfo
PerturbationType = perturbation_module.PerturbationType
generate_text = perturbation_module.generate_text

console = Console()


class Level(StrEnum):
    name_only = "name_only"
    name_target = "name_target"
    full = "full"


def _type_name(value: Any) -> str:
    if hasattr(value, "__name__"):
        return value.__name__
    return str(value).replace("typing.", "")


def _field_schema() -> list[dict[str, str]]:
    return [
        {
            "name": field.name,
            "type": _type_name(field.type),
            "default": str(field.default),
        }
        for field in fields(PerturbationInfo)
    ]


def _info_to_dict(info: PerturbationInfo) -> dict[str, str]:
    data: dict[str, str] = {}
    for field in fields(PerturbationInfo):
        value = getattr(info, field.name)
        data[field.name] = str(value)
    return data


def _build_examples() -> dict[PerturbationType, PerturbationInfo]:
    return {
        PerturbationType.COMPOUND: PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            broad_sample="BRD-K00000001",
            pert_iname="Aloxistatin",
            target_list="CTSL",
            smiles="CC(CC)C=O",
            pubchem_cid="12345",
            moa="Cysteine protease inhibitor",
        ),
        PerturbationType.CRISPR: PerturbationInfo(
            pert_type=PerturbationType.CRISPR,
            broad_sample="CRISPR-TP53-1",
            pert_iname="TP53",
            gene="TP53",
            protein_name="Tumor protein p53",
            moa="Tumor suppressor",
            go_terms="apoptotic process (GO:0006915)",
        ),
        PerturbationType.ORF: PerturbationInfo(
            pert_type=PerturbationType.ORF,
            broad_sample="ORF-BRCA1-1",
            pert_iname="BRCA1",
            gene="BRCA1",
            protein_name="Breast cancer type 1 susceptibility protein",
            moa="DNA repair regulator",
        ),
        PerturbationType.NEGCON: PerturbationInfo(
            pert_type=PerturbationType.NEGCON,
            control_type="negcon",
        ),
        PerturbationType.POSCON: PerturbationInfo(
            pert_type=PerturbationType.POSCON,
            control_type="poscon_diverse",
        ),
        PerturbationType.UNKNOWN: PerturbationInfo(
            pert_type=PerturbationType.UNKNOWN,
            broad_sample="MYSTERY-001",
        ),
    }


def _load_index(
    config_path: Path,
    metadata_dir: Path | None,
    batch: str | None,
) -> Any:
    try:
        metadata_module = _load_module("morphoclip.data.metadata", DATA_ROOT / "metadata.py")
        MetadataIndex = metadata_module.MetadataIndex
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Metadata lookup requires the project dependencies. "
            "Run the script via `uv run poe inspect-perturbation ...` or install deps first."
        ) from exc

    if metadata_dir is not None:
        return MetadataIndex.from_directory(metadata_dir, batch=batch)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading configs/dataset.yml requires PyYAML. "
            "Run the script via `uv run poe inspect-perturbation ...` or install deps first."
        ) from exc

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)["cpjump"]

    config_metadata_dir = Path(config["local"]["metadata"])
    config_batch = batch or config.get("batch")
    return MetadataIndex.from_directory(config_metadata_dir, batch=config_batch)


def _render_text_output(
    schema: list[dict[str, str]],
    examples: list[dict[str, Any]],
    lookup_result: dict[str, Any] | None,
) -> str:
    lines = [
        "PerturbationType values:",
        f"  {', '.join(member.value for member in PerturbationType)}",
        "",
        "PerturbationInfo fields:",
    ]

    for field in schema:
        lines.append(f"  - {field['name']}: {field['type']} (default={field['default']!s})")

    lines.append("")
    lines.append("Template examples:")
    for example in examples:
        lines.append(f"  - {example['pert_type']}:")
        for level, text in example["text_by_level"].items():
            lines.append(f"      {level}: {text}")

    if lookup_result is not None:
        lines.extend(
            [
                "",
                f"Metadata lookup: plate={lookup_result['plate']} well={lookup_result['well']}",
                f"  text ({lookup_result['level']}): {lookup_result['text']}",
            ]
        )
        for key, value in lookup_result["info"].items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def main(
    config: Annotated[Path, typer.Option(help="Dataset config YAML.")] = Path(
        "configs/dataset.yml"
    ),
    metadata_dir: Annotated[
        Path | None, typer.Option(help="Metadata directory (default: from config).")
    ] = None,
    batch: Annotated[str | None, typer.Option(help="Batch name (default: from config).")] = None,
    plate: Annotated[str | None, typer.Option(help="Plate barcode, e.g. BR00116991.")] = None,
    well: Annotated[str | None, typer.Option(help="Well position, e.g. A01.")] = None,
    level: Annotated[
        Level, typer.Option(help="Text generation level used for examples and metadata lookup.")
    ] = Level.full,
    example_type: Annotated[
        PerturbationType | None,
        typer.Option(help="Restrict template output to one perturbation type."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json/--no-json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Inspect MorphoCLIP perturbation types, fields, and metadata-backed examples."""
    if (plate is None) != (well is None):
        raise typer.BadParameter("--plate and --well must be provided together.")

    schema = _field_schema()
    example_infos = _build_examples()

    selected_examples: list[dict[str, Any]] = []
    for pert_type, info in example_infos.items():
        if example_type is not None and pert_type != example_type:
            continue
        selected_examples.append(
            {
                "pert_type": pert_type.value,
                "info": _info_to_dict(info),
                "text_by_level": {
                    text_level: generate_text(info, level=text_level)
                    for text_level in ("name_only", "name_target", "full")
                },
            }
        )

    lookup_result: dict[str, Any] | None = None
    if plate is not None and well is not None:
        try:
            index = _load_index(config, metadata_dir, batch)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        info = index.lookup(plate, well)
        lookup_result = {
            "plate": plate,
            "well": well.upper(),
            "level": level,
            "text": generate_text(info, level=level),
            "info": _info_to_dict(info),
        }

    payload = {
        "enum_values": [member.value for member in PerturbationType],
        "schema": schema,
        "examples": selected_examples,
        "lookup": lookup_result,
    }

    if json_output:
        print(json.dumps(payload, indent=2))
        return

    console.print(_render_text_output(schema, selected_examples, lookup_result), markup=False)


if __name__ == "__main__":
    typer.run(main)
