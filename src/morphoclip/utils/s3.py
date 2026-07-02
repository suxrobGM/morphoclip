"""S3 transfer utilities for AWS CLI and rclone backends."""

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

DEFAULT_RCLONE_REMOTE = ":s3,provider=AWS,region=us-east-1,no_check_bucket=true:"

console = Console()


def build_s3_uri(endpoint: str, path_template: str, batch: str) -> str:
    """Build a full S3 URI by joining an endpoint with a formatted path.

    Args:
        endpoint: Base S3 URI (e.g. ``s3://cellpainting-gallery/cpg0000``).
        path_template: Path segment with an optional ``{batch}`` placeholder
            (e.g. ``images/{batch}/images``).
        batch: Value substituted into *path_template*.

    Returns:
        The assembled S3 URI.
    """
    return f"{endpoint.rstrip('/')}/{path_template.format(batch=batch)}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an S3 URI into its bucket name and object prefix.

    Args:
        uri: An S3 URI of the form ``s3://bucket/prefix/path``.

    Returns:
        A ``(bucket, prefix)`` tuple.

    Raises:
        ValueError: If *uri* does not start with ``s3://``.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected S3 URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def choose_backend(requested: str) -> str:
    """Select an S3 transfer backend, falling back to what is installed.

    Args:
        requested: One of ``"awscli"``, ``"rclone"``, or ``"auto"``.
            When ``"auto"``, the first available backend is used
            (preferring ``awscli``).

    Returns:
        The resolved backend name (``"awscli"`` or ``"rclone"``).

    Raises:
        RuntimeError: If ``"auto"`` is requested and neither tool is found
            on ``PATH``.
    """
    if requested in {"awscli", "rclone"}:
        return requested
    if shutil.which("aws") is not None:
        return "awscli"
    if shutil.which("rclone") is not None:
        return "rclone"
    raise RuntimeError("No supported downloader found. Install aws CLI or rclone.")


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    """Print and optionally execute a shell command.

    The command is always logged to the console. Execution is skipped
    when *dry_run* is ``True``.

    Args:
        cmd: Command and arguments to run (passed to :func:`subprocess.run`).
        dry_run: If ``True``, only print the command without executing it.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            status.
    """
    console.print(f"  [dim]{' '.join(cmd)}[/dim]")
    if not dry_run:
        subprocess.run(cmd, check=True)


def sync_s3_path(
    s3_uri: str,
    dest: Path,
    *,
    backend: str,
    no_sign_request: bool = True,
    rclone_remote: str = DEFAULT_RCLONE_REMOTE,
    dry_run: bool = False,
) -> None:
    """Download an S3 prefix to a local directory.

    Creates *dest* if it does not already exist, then delegates to
    ``aws s3 sync`` or ``rclone copy`` depending on *backend*.

    Args:
        s3_uri: Source S3 URI (e.g. ``s3://bucket/prefix``).
        dest: Local directory to sync files into.
        backend: Transfer tool — ``"awscli"`` or ``"rclone"``.
        no_sign_request: If ``True``, pass ``--no-sign-request`` to the
            AWS CLI (for public buckets). Ignored when using rclone.
        rclone_remote: Rclone remote specification. Defaults to an
            anonymous AWS S3 remote.
        dry_run: If ``True``, log the command without executing it.

    Raises:
        ValueError: If *backend* is not ``"awscli"`` or ``"rclone"``.
        subprocess.CalledProcessError: If the transfer command fails.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if backend == "awscli":
        cmd = ["aws", "s3", "sync", s3_uri, str(dest)]
        if no_sign_request:
            cmd.append("--no-sign-request")
        run_cmd(cmd, dry_run=dry_run)
    elif backend == "rclone":
        bucket, prefix = parse_s3_uri(s3_uri)
        remote_path = f"{rclone_remote}{bucket}/{prefix}"
        cmd = [
            "rclone",
            "copy",
            remote_path,
            str(dest),
            "--progress",
            "--transfers",
            "8",
            "--checkers",
            "16",
        ]
        run_cmd(cmd, dry_run=dry_run)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
