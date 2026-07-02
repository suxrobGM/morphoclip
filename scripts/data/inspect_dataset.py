"""Inspect MorphoCLIP dataset indexing, samples, and split behavior.

Examples:
    uv run poe inspect-dataset --demo --sample-index 0
    uv run poe inspect-dataset --demo --collate-preview 3 \
        --split-strategy cpjump1_official_representation
    uv run poe inspect-dataset --feature-dir data/features --plates BR00116991 BR00117000
    uv run poe inspect-dataset --list-groups cellclip_cpjump_style cpjump1_official_gene_compound
"""

import argparse
import importlib.util
import inspect
import json
import statistics
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark import splits as benchmark_splits

PROJECT_ROOT = Path(__file__).parent.parent
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


def _load_project_modules() -> dict[str, Any]:
    _ensure_namespace_packages()
    try:
        perturbation = _load_module("morphoclip.data.perturbation", DATA_ROOT / "perturbation.py")
        image_loader = _load_module("morphoclip.data.image_loader", DATA_ROOT / "image_loader.py")
        metadata = _load_module("morphoclip.data.metadata", DATA_ROOT / "metadata.py")
        dataset = _load_module("morphoclip.data.dataset", DATA_ROOT / "dataset.py")
        splits = _load_module("morphoclip.data.splits", DATA_ROOT / "splits.py")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dataset inspection requires the project dependencies. "
            "Run via `uv run poe inspect-dataset ...` or install the dependencies first."
        ) from exc

    return {
        "perturbation": perturbation,
        "image_loader": image_loader,
        "metadata": metadata,
        "dataset": dataset,
        "splits": splits,
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading configs/dataset.yml requires PyYAML. "
            "Run via `uv run poe inspect-dataset ...` or install the dependencies first."
        ) from exc

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["cpjump"]


def _normalize_plates(raw_plates: list[str], extract_plate_barcode: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for plate in raw_plates:
        barcode = extract_plate_barcode(plate)
        if barcode not in seen:
            normalized.append(barcode)
            seen.add(barcode)
    return normalized


def _auto_detect_plates(feature_dir: Path) -> list[str]:
    if not feature_dir.exists():
        return []
    return sorted(path.name for path in feature_dir.iterdir() if path.is_dir())


def _create_demo_dataset(
    root: Path,
    mode: str,
    row_col_from_well: Any,
    torch: Any,
) -> list[str]:
    demo_layout = {
        "BR00116991": {"A01": 2, "A02": 1, "A03": 2, "A04": 3},
        "BR00117000": {"A01": 2, "A02": 2, "A03": 1, "A04": 2},
    }

    for plate, wells in demo_layout.items():
        plate_dir = root / plate
        plate_dir.mkdir(parents=True, exist_ok=True)
        for well, num_sites in wells.items():
            row, col = row_col_from_well(well)
            for field in range(1, num_sites + 1):
                if mode == "features":
                    tensor = torch.randn(5, 384)
                else:
                    tensor = torch.randn(5, 64, 64)
                torch.save(tensor, plate_dir / f"r{row:02d}c{col:02d}f{field:02d}.pt")

    return sorted(demo_layout.keys())


class _IndexOnlyDataset:
    def __init__(self, metadata: Any, index_entries: list[tuple[str, str, list[Path]]]) -> None:
        self._metadata = metadata
        self._index_entries = index_entries

    @property
    def metadata(self) -> Any:
        return self._metadata

    @property
    def index_entries(self) -> list[tuple[str, str, list[Path]]]:
        return self._index_entries

    def __len__(self) -> int:
        return len(self._index_entries)


def _build_metadata_only_dataset(
    metadata: Any,
    plates: list[str],
    exclude_controls: bool,
    pert_types: set[Any] | None,
) -> _IndexOnlyDataset:
    index_entries: list[tuple[str, str, list[Path]]] = []

    for plate in plates:
        for well in metadata.wells_for_plate(plate):
            info = metadata.lookup(plate, well)

            if exclude_controls and info.pert_type.value in {"negcon", "poscon"}:
                continue

            if pert_types is not None and info.pert_type not in pert_types:
                continue

            index_entries.append((plate, well, []))

    return _IndexOnlyDataset(metadata=metadata, index_entries=index_entries)


def _summarize_dataset(dataset: Any, extract_plate_barcode: Any) -> dict[str, Any]:
    entries = dataset.index_entries
    plates = [plate for plate, _, _ in entries]
    wells = [well for _, well, _ in entries]
    site_counts = [len(paths) for _, _, paths in entries]

    pert_counter: Counter[str] = Counter()
    for plate, well, _ in entries:
        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        pert_counter[info.pert_type.value] += 1

    summary: dict[str, Any] = {
        "num_wells": len(entries),
        "num_unique_plates": len(set(plates)),
        "num_total_sites": sum(site_counts),
        "plates": sorted(set(plates)),
        "per_plate_wells": dict(sorted(Counter(plates).items())),
        "per_perturbation_type_wells": dict(sorted(pert_counter.items())),
        "first_entries": [
            {
                "plate": plate,
                "well": well,
                "num_sites": len(paths),
                "site_files": [path.name for path in paths[:3]],
            }
            for plate, well, paths in entries[:5]
        ],
    }

    if site_counts:
        summary["sites_per_well"] = {
            "min": min(site_counts),
            "median": statistics.median(site_counts),
            "mean": round(statistics.fmean(site_counts), 3),
            "max": max(site_counts),
        }

    if wells:
        summary["first_wells"] = wells[:5]

    return summary


def _summarize_sample(sample: Any) -> dict[str, Any]:
    return {
        "plate": sample.plate,
        "well": sample.well,
        "text": sample.text,
        "pert_type": sample.pert_info.pert_type.value,
        "broad_sample": sample.pert_info.broad_sample,
        "features_shape": list(sample.features.shape),
        "features_dtype": str(sample.features.dtype),
    }


def _summarize_collated_batch(batch: dict[str, Any]) -> dict[str, Any]:
    site_mask = batch["site_mask"]
    return {
        "features_shape": list(batch["features"].shape),
        "site_mask_shape": list(site_mask.shape),
        "site_counts": [int(mask_row.sum().item()) for mask_row in site_mask],
        "plates": batch["plates"],
        "wells": batch["wells"],
    }


def _summarize_splits(train: Any, val: Any, test: Any) -> dict[str, Any]:
    return {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_indices_preview": list(train.indices[:5]),
        "val_indices_preview": list(val.indices[:5]),
        "test_indices_preview": list(test.indices[:5]),
    }


def _summarize_groups(
    dataset: Any,
    strategy: str,
    build_split_groups: Any,
    extract_plate_barcode: Any,
    group_limit: int,
    member_limit: int,
    show_all: bool,
) -> dict[str, Any]:
    group_map = build_split_groups(dataset, strategy=strategy)
    sorted_items = sorted(group_map.items(), key=lambda item: item[0])
    group_sizes = [len(indices) for _, indices in sorted_items]

    groups: list[dict[str, Any]] = []
    selected_items = sorted_items if show_all else sorted_items[:group_limit]
    for group_key, indices in selected_items:
        members: list[dict[str, Any]] = []
        selected_indices = indices if show_all else indices[:member_limit]
        for idx in selected_indices:
            plate, well, paths = dataset.index_entries[idx]
            barcode = extract_plate_barcode(plate)
            info = dataset.metadata.lookup(barcode, well)
            members.append(
                {
                    "index": idx,
                    "plate": plate,
                    "well": well,
                    "num_sites": len(paths),
                    "pert_type": info.pert_type.value,
                    "broad_sample": info.broad_sample,
                }
            )

        groups.append(
            {
                "group_key": group_key,
                "num_entries": len(indices),
                "member_count_shown": len(members),
                "members": members,
            }
        )

    summary: dict[str, Any] = {
        "strategy": strategy,
        "num_groups": len(sorted_items),
        "groups_shown": len(groups),
        "group_size_stats": {},
        "groups": groups,
    }

    if group_sizes:
        summary["group_size_stats"] = {
            "min": min(group_sizes),
            "median": statistics.median(group_sizes),
            "mean": round(statistics.fmean(group_sizes), 3),
            "max": max(group_sizes),
        }

    return summary


def _render_text(payload: dict[str, Any]) -> str:
    input_plates = (
        ", ".join(payload["inputs"]["plates"]) if payload["inputs"]["plates"] else "(none)"
    )
    lines = [
        "MorphoCLIP dataset inspection",
        "",
        "Class schema:",
        f"  MorphoCLIPSample slots: {', '.join(payload['class_schema']['sample_slots'])}",
        f"  MorphoCLIPDataset.__init__: {payload['class_schema']['dataset_init_signature']}",
        f"  collate_fn returns: {', '.join(payload['class_schema']['collate_output_keys'])}",
        "",
        "Resolved inputs:",
        f"  mode: {payload['inputs']['mode']}",
        f"  feature_dir: {payload['inputs']['feature_dir']}",
        f"  metadata_dir: {payload['inputs']['metadata_dir']}",
        f"  batch: {payload['inputs']['batch']}",
        f"  plates: {input_plates}",
        f"  demo: {payload['inputs']['demo']}",
        f"  inspection basis: {payload['inputs']['inspection_basis']}",
        f"  experiment metadata: {payload['inputs']['experiment_metadata_path']}",
    ]

    dataset_summary = payload["dataset_summary"]
    lines.extend(
        [
            "",
            "Dataset summary:",
            f"  wells: {dataset_summary['num_wells']}",
            f"  unique plates: {dataset_summary['num_unique_plates']}",
            f"  total sites: {dataset_summary['num_total_sites']}",
        ]
    )

    if "sites_per_well" in dataset_summary:
        stats = dataset_summary["sites_per_well"]
        lines.append(
            "  sites per well: "
            f"min={stats['min']} median={stats['median']} mean={stats['mean']} max={stats['max']}"
        )

    lines.append(f"  per plate wells: {dataset_summary['per_plate_wells']}")
    lines.append(f"  per perturbation type wells: {dataset_summary['per_perturbation_type_wells']}")

    if dataset_summary["first_entries"]:
        lines.append("  first entries:")
        for entry in dataset_summary["first_entries"]:
            lines.append(
                f"    {entry['plate']} {entry['well']} -> {entry['num_sites']} sites "
                f"({', '.join(entry['site_files'])})"
            )

    if payload["sample_preview"] is not None:
        sample = payload["sample_preview"]
        lines.extend(
            [
                "",
                f"Sample preview: index={payload['sample_index']}",
                f"  {sample['plate']} {sample['well']} [{sample['pert_type']}]",
                f"  features: shape={sample['features_shape']} dtype={sample['features_dtype']}",
                f"  broad_sample: {sample['broad_sample']}",
                f"  text: {sample['text']}",
            ]
        )

    if payload["collate_preview"] is not None:
        collate = payload["collate_preview"]
        lines.extend(
            [
                "",
                "Collate preview:",
                f"  features: {collate['features_shape']}",
                f"  site_mask: {collate['site_mask_shape']}",
                f"  site counts: {collate['site_counts']}",
                f"  wells: {collate['wells']}",
            ]
        )

    if payload["split_summary"] is not None:
        split_summary = payload["split_summary"]
        lines.extend(
            [
                "",
                f"Split summary ({payload['split_strategy']}):",
                f"  train={split_summary['train']} val={split_summary['val']} "
                f"test={split_summary['test']}",
                f"  train preview={split_summary['train_indices_preview']}",
                f"  val preview={split_summary['val_indices_preview']}",
                f"  test preview={split_summary['test_indices_preview']}",
            ]
        )

    if payload["group_summaries"]:
        lines.append("")
        lines.append("Grouping inspection:")
        for group_summary in payload["group_summaries"]:
            lines.append(
                f"  {group_summary['strategy']}: {group_summary['num_groups']} groups "
                f"(showing {group_summary['groups_shown']})"
            )
            if group_summary["group_size_stats"]:
                stats = group_summary["group_size_stats"]
                lines.append(
                    "    group sizes: "
                    f"min={stats['min']} median={stats['median']} "
                    f"mean={stats['mean']} max={stats['max']}"
                )
            for group in group_summary["groups"]:
                lines.append(f"    {group['group_key']} -> {group['num_entries']} entries")
                for member in group["members"]:
                    lines.append(
                        f"      idx={member['index']} {member['plate']} {member['well']} "
                        f"sites={member['num_sites']} pert_type={member['pert_type']} "
                        f"broad_sample={member['broad_sample']}"
                    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MorphoCLIPDataset indexing, sample loading, and split behavior."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yml"))
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--plates", nargs="*", default=None)
    parser.add_argument("--mode", choices=["features", "tensors"], default="features")
    parser.add_argument(
        "--text-level", choices=["name_only", "name_target", "full"], default="full"
    )
    parser.add_argument("--exclude-controls", action="store_true")
    parser.add_argument("--pert-types", nargs="*", default=None)
    parser.add_argument("--max-sites-per-well", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--collate-preview", type=int, default=0)
    parser.add_argument(
        "--split-strategy",
        choices=[
            "cpjump1_official_representation",
            "cpjump1_official_gene_compound",
            "cellclip_cpjump_style",
        ],
        default=None,
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--list-groups",
        nargs="+",
        choices=[
            "cpjump1_official_representation",
            "cpjump1_official_gene_compound",
            "cellclip_cpjump_style",
        ],
        default=None,
        help="List actual grouping keys and members for the selected strategies.",
    )
    parser.add_argument(
        "--group-limit",
        type=int,
        default=20,
        help="Maximum number of groups to print per strategy.",
    )
    parser.add_argument(
        "--group-member-limit",
        type=int,
        default=10,
        help="Maximum number of members to print inside each group.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all groups and all members for --list-groups.",
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        modules = _load_project_modules()
    except RuntimeError as exc:
        parser.error(str(exc))

    perturbation = modules["perturbation"]
    metadata_module = modules["metadata"]
    dataset_module = modules["dataset"]

    extract_plate_barcode = perturbation.extract_plate_barcode
    row_col_from_well = perturbation.row_col_from_well
    PerturbationType = perturbation.PerturbationType
    MetadataIndex = metadata_module.MetadataIndex
    MorphoCLIPDataset = dataset_module.MorphoCLIPDataset
    MorphoCLIPSample = dataset_module.MorphoCLIPSample
    collate_fn = dataset_module.collate_fn

    create_splits = benchmark_splits.create_splits
    build_split_groups = benchmark_splits.build_split_groups
    metadata_path = benchmark_splits.METADATA_PATH

    config = _load_config(args.config)
    local_config = config.get("local", {})
    default_feature_dir = Path(local_config["features" if args.mode == "features" else "tensors"])
    feature_dir = args.feature_dir or default_feature_dir
    metadata_dir = args.metadata_dir or Path(local_config["metadata"])
    batch = args.batch or config.get("batch")

    if args.plates is not None:
        plates = _normalize_plates(args.plates, extract_plate_barcode)
    else:
        plates = _auto_detect_plates(feature_dir)
        if not plates and not feature_dir.exists():
            plates = _normalize_plates(config.get("plates", []), extract_plate_barcode)

    if args.pert_types is not None:
        invalid_types = [
            name for name in args.pert_types if name not in {p.value for p in PerturbationType}
        ]
        if invalid_types:
            parser.error(f"Unknown perturbation type(s): {', '.join(invalid_types)}")
        pert_types = {PerturbationType(name) for name in args.pert_types}
    else:
        pert_types = None

    payload: dict[str, Any] = {
        "class_schema": {
            "sample_slots": list(MorphoCLIPSample.__slots__),
            "dataset_init_signature": str(inspect.signature(MorphoCLIPDataset.__init__)),
            "collate_output_keys": [
                "features",
                "site_mask",
                "text",
                "plates",
                "wells",
                "pert_info",
            ],
        },
        "inputs": {
            "mode": args.mode,
            "feature_dir": str(feature_dir),
            "metadata_dir": str(metadata_dir),
            "batch": batch,
            "plates": plates,
            "demo": args.demo,
            "inspection_basis": "feature_index",
            "experiment_metadata_path": str(metadata_path),
        },
        "dataset_summary": {},
        "sample_index": args.sample_index,
        "sample_preview": None,
        "collate_preview": None,
        "split_strategy": args.split_strategy,
        "split_summary": None,
        "group_summaries": [],
    }

    with tempfile.TemporaryDirectory(prefix="inspect_dataset_") as tmpdir:
        torch = dataset_module.torch
        if args.demo:
            feature_dir = Path(tmpdir)
            plates = _create_demo_dataset(feature_dir, args.mode, row_col_from_well, torch)
            payload["inputs"]["feature_dir"] = str(feature_dir)
            payload["inputs"]["plates"] = plates

        metadata = MetadataIndex.from_directory(metadata_dir, batch=batch)
        dataset = MorphoCLIPDataset(
            feature_dir=feature_dir,
            metadata=metadata,
            plates=plates,
            mode=args.mode,
            text_level=args.text_level,
            exclude_controls=args.exclude_controls,
            pert_types=pert_types,
            max_sites_per_well=args.max_sites_per_well,
        )

        inspection_dataset = dataset
        if len(dataset) == 0 and (args.split_strategy is not None or args.list_groups is not None):
            if not plates:
                plates = _normalize_plates(config.get("plates", []), extract_plate_barcode)
            if not plates:
                plates = metadata.plates()

            payload["inputs"]["plates"] = plates
            payload["inputs"]["inspection_basis"] = "metadata_only"
            inspection_dataset = _build_metadata_only_dataset(
                metadata=metadata,
                plates=plates,
                exclude_controls=args.exclude_controls,
                pert_types=pert_types,
            )

        payload["dataset_summary"] = _summarize_dataset(inspection_dataset, extract_plate_barcode)

        if args.sample_index is not None:
            if not (0 <= args.sample_index < len(dataset)):
                parser.error(f"--sample-index must be in [0, {len(dataset) - 1}] for this dataset.")
            payload["sample_preview"] = _summarize_sample(dataset[args.sample_index])

        if args.collate_preview > 0:
            batch_size = min(args.collate_preview, len(dataset))
            if batch_size == 0:
                parser.error("--collate-preview requested, but the dataset is empty.")
            samples = [dataset[i] for i in range(batch_size)]
            payload["collate_preview"] = _summarize_collated_batch(collate_fn(samples))

        if args.split_strategy is not None:
            if len(inspection_dataset) == 0:
                parser.error("--split-strategy requested, but the dataset is empty.")
            train, val, test = create_splits(
                inspection_dataset,
                strategy=args.split_strategy,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
            payload["split_summary"] = _summarize_splits(train, val, test)

        if args.list_groups is not None:
            if len(inspection_dataset) == 0:
                parser.error("--list-groups requested, but the dataset is empty.")
            payload["group_summaries"] = [
                _summarize_groups(
                    inspection_dataset,
                    strategy=strategy,
                    build_split_groups=build_split_groups,
                    extract_plate_barcode=extract_plate_barcode,
                    group_limit=args.group_limit,
                    member_limit=args.group_member_limit,
                    show_all=args.all,
                )
                for strategy in args.list_groups
            ]

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(_render_text(payload))


if __name__ == "__main__":
    main()
