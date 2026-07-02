"""Hugging Face transfer helpers for tarred DINOv3 feature archives.

Presentation-free I/O (no Rich/console) so the CLI command bodies stay in charge
of progress rendering.
"""

import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, login

DEFAULT_REPO_ID = "suxrobgm/cpjump1-dinov3-features"


def list_local_archives(features_dir: Path) -> list[Path]:
    """Return sorted list of .tar.gz archives in the directory."""
    return sorted(features_dir.glob("*.tar.gz"))


def list_repo_archives(api: HfApi, repo_id: str) -> list[str]:
    """List all .tar.gz files in the HF dataset repo."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".tar.gz") and not f.startswith("."))


def is_plate_extracted(plate_dir: Path) -> bool:
    """Whether a plate directory already holds extracted ``.pt`` features."""
    return plate_dir.is_dir() and any(plate_dir.glob("*.pt"))


def partition_pending_archives(
    archives: list[str], output_dir: Path
) -> tuple[list[str], int]:
    """Split repo archives into (pending, already-extracted-count)."""
    pending: list[str] = []
    skipped = 0
    for archive in archives:
        plate_name = archive.removesuffix(".tar.gz")
        if is_plate_extracted(output_dir / plate_name):
            skipped += 1
        else:
            pending.append(archive)
    return pending, skipped


def download_and_extract_archive(
    *, api: HfApi, repo_id: str, filename: str, output_dir: Path, skip_extract: bool
) -> str:
    """Download a single archive and extract it. Returns a status string."""
    plate_name = filename.removesuffix(".tar.gz")
    plate_dir = output_dir / plate_name
    if is_plate_extracted(plate_dir):
        return f"{plate_name} (skipped, already extracted)"

    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    if skip_extract:
        return f"{plate_name} (downloaded)"

    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    return f"{plate_name} (done)"


def upload_folder(
    features_dir: Path,
    *,
    repo_id: str,
    revision: str | None,
    num_workers: int,
) -> None:
    """Log in, ensure the dataset repo exists, and upload the folder (resumable)."""
    login()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(features_dir),
        revision=revision,
        num_workers=num_workers,
    )
