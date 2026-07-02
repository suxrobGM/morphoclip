"""Extract DINOv3 features from downloaded CPJUMP1 plates.

Processes plates one at a time. For each plate:
1. Load all sites from the plate's Images/ directory
2. Extract per-channel CLS tokens from frozen DINOv3
3. Save features to data/features/{plate}/
4. Optionally save preprocessed tensors to data/tensors/{plate}/

Usage:
    uv run poe extract-features
    uv run poe extract-features --plate BR00116991
    uv run poe extract-features --verify-only
    uv run poe extract-features --visualize
    uv run poe extract-features --plate BR00116991 --visualize-only
    uv run poe extract-features --plate BR00116991 --visualize --visualize-n 8
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.data.feature_extractor import (  # noqa: E402
    extract_plate_features,
    verify_plate_features,
)
from morphoclip.data.image_loader import discover_sites, load_site_as_tensor  # noqa: E402
from morphoclip.data.perturbation import extract_plate_barcode  # noqa: E402
from morphoclip.data.visualize import save_site_comparison  # noqa: E402

console = Console()


def _clear_pt_files(directory: Path) -> int:
    """Remove saved ``.pt`` files from a directory if it exists."""
    if not directory.exists():
        return 0

    removed = 0
    for path in directory.glob("*.pt"):
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv3 features from CPJUMP1 plates.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yml"))
    parser.add_argument("--plate", type=str, help="Extract a specific plate only.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the vision backbone model ID used for site feature extraction.",
    )
    parser.add_argument(
        "--compressed-root",
        type=Path,
        default=None,
        help="Override the compressed image root instead of the dataset config value.",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Override the output root for extracted feature .pt files.",
    )
    parser.add_argument(
        "--tensors-root",
        type=Path,
        default=None,
        help="Override the output root for saved resized tensors.",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't extract.")
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g. cuda, cpu)."
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--no-tensors", action="store_true", help="Skip saving resized tensors.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save channel grid and composite PNGs for sample sites.",
    )
    parser.add_argument(
        "--visualize-n",
        type=int,
        default=4,
        help="Number of sample sites to visualize per plate (default: 4).",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualization images, skip extraction.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)["cpjump"]

    extraction = config.get("extraction", {})
    local = config.get("local", {})
    model_name = args.model_name or extraction.get(
        "model", "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    device = args.device or extraction.get("device", "auto")
    batch_size = args.batch_size or extraction.get("batch_size", 32)

    compressed_root = args.compressed_root or Path(
        local.get(
            "compressed_images",
            config.get("compression", {})
            .get("default", {})
            .get("output_root", "data/raw_compressed"),
        )
    )
    features_root = args.features_root or Path(local.get("features", "data/features"))
    tensors_root = args.tensors_root or Path(local.get("tensors", "data/tensors"))

    # Determine which plates to process
    plates = config.get("plates", [])
    if args.plate:
        plates = [p for p in plates if extract_plate_barcode(p) == args.plate or p == args.plate]
        if not plates:
            # Try the barcode directly as a plate directory name
            plates = [args.plate]

    console.rule("[bold blue]DINOv3 Feature Extraction")
    console.print(f"  Model:      {model_name}")
    console.print(f"  Device:     {device}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Plates:     {len(plates)}")

    for plate_name in plates:
        barcode = extract_plate_barcode(plate_name)

        # Find the image directory
        batch = config.get("batch", "")
        image_dir = compressed_root / batch / plate_name / "Images"
        if not image_dir.exists():
            # Try without the batch prefix
            image_dir = compressed_root / plate_name / "Images"
        if not image_dir.exists():
            console.print(f"\n[bold red]Image directory not found: {image_dir}")
            continue

        feature_dir = features_root / barcode
        tensor_dir = tensors_root / barcode

        if args.verify_only:
            console.print(f"\n[bold]Verifying [cyan]{barcode}[/cyan]...")
            extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
            console.print(f"  Extracted: {extracted}/{expected}")
            if missing:
                console.print(f"  [red]Missing {len(missing)} sites[/red]")
            else:
                console.print("  [green]All sites extracted[/green]")
            continue

        # Visualization (before or instead of extraction)
        if args.visualize or args.visualize_only:
            vis_dir = Path("data/visualizations") / barcode
            sites = discover_sites(image_dir)
            sample_keys = sorted(sites.keys(), key=str)[: args.visualize_n]
            console.print(
                f"\n[bold]Visualizing [cyan]{barcode}[/cyan] ({len(sample_keys)} sample sites)..."
            )
            for key in sample_keys:
                site_tensor = load_site_as_tensor(sites[key], resize=384)

                # Load CLS features if they exist
                feat_path = feature_dir / f"r{key.row:02d}c{key.col:02d}f{key.field:02d}.pt"
                cls_features = None
                if feat_path.exists():
                    cls_features = torch.load(feat_path, weights_only=True)

                cmp_path = save_site_comparison(
                    site_tensor,
                    key,
                    vis_dir,
                    cls_features=cls_features,
                )
                console.print(f"  {key}: [dim]{cmp_path}[/dim]")
                if cls_features is not None:
                    console.print(f"         [dim]CLS features: {tuple(cls_features.shape)}[/dim]")
                else:
                    console.print(
                        "         [yellow]No CLS features found (run extraction first)[/yellow]"
                    )
            console.print(f"  [green]Saved {len(sample_keys)} images to {vis_dir}[/green]")
            if args.visualize_only:
                continue

        console.print(f"\n[bold]Processing plate [cyan]{barcode}[/cyan]...")
        console.print(f"  Images:   {image_dir}")
        console.print(f"  Features: {feature_dir}")

        extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
        if extracted == expected and not missing:
            console.print(
                f"  [yellow]Skipping[/yellow] existing complete batch ({extracted}/{expected})"
            )
            continue

        if extracted > 0:
            console.print(
                "  [yellow]Incomplete output detected[/yellow] "
                f"({extracted}/{expected}); clearing and re-extracting batch"
            )
            removed_features = _clear_pt_files(feature_dir)
            console.print(f"  Cleared {removed_features} existing feature files")
            if not args.no_tensors:
                removed_tensors = _clear_pt_files(tensor_dir)
                if removed_tensors:
                    console.print(f"  Cleared {removed_tensors} existing tensor files")

        saved = extract_plate_features(
            image_dir=image_dir,
            output_dir=feature_dir,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            save_tensors=not args.no_tensors,
            tensor_output_dir=tensor_dir if not args.no_tensors else None,
        )
        console.print(f"  [green]Saved {len(saved)} feature files[/green]")

    console.print("\n[bold green]Done.")


if __name__ == "__main__":
    main()
