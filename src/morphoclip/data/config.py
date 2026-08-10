"""Schema for the ``cpjump`` section of ``configs/dataset.yml``.

Seven call sites used to read this file with a bare
``yaml.safe_load(f)["cpjump"]`` and reach into nested dicts with their own
defaults, so the same key had a different fallback depending on who asked.
"""

from pathlib import Path

from pydantic import Field

from morphoclip.config import StrictModel, load_yaml_mapping

DEFAULT_DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_RCLONE_REMOTE = ":s3,provider=AWS,region=us-east-1,no_check_bucket=true:"


class LocalPaths(StrictModel):
    """Where each stage of the pipeline writes, relative to the project root."""

    raw_images: Path = Path("data/raw")
    metadata: Path = Path("data/metadata")
    features: Path = Path("data/features")
    tensors: Path = Path("data/tensors")
    compressed_images: Path = Path("data/raw_compressed")


class ExtractionConfig(StrictModel):
    """DINOv3 settings for CLS-token extraction."""

    model: str = DEFAULT_DINO_MODEL
    device: str = "auto"
    batch_size: int = Field(default=32, gt=0)


class FetchConfig(StrictModel):
    """How plates are downloaded from the Cell Painting Gallery bucket."""

    backend: str = "auto"
    on_existing_plate: str = "ask"
    aws_no_sign_request: bool = True
    rclone_remote: str = DEFAULT_RCLONE_REMOTE


class CPJumpConfig(StrictModel):
    """The ``cpjump`` section: S3 source, plate list, local paths, and stage settings.

    The four S3 fields describe the public Cell Painting Gallery layout rather
    than a user choice, so they default. That keeps a config written for
    local-only work valid, since nothing but ``morphoclip data fetch`` reads them.
    """

    batch: str
    endpoint: str = "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
    images: str = "images/{batch}/images"
    metadata: str = "workspace/metadata/platemaps/{batch}"
    external_metadata: str = "workspace/metadata/external_metadata"
    plates: list[str] = Field(default_factory=list)
    local: LocalPaths = Field(default_factory=LocalPaths)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)

    def s3_uri(self, template: str) -> str:
        """Expand a ``{batch}`` path template against the endpoint."""
        return f"{self.endpoint.rstrip('/')}/{template.format(batch=self.batch).strip('/')}"


def load_dataset_config(path: str | Path) -> CPJumpConfig:
    """Load the ``cpjump`` section of a dataset config YAML.

    Raises:
        ValueError: If the file has no ``cpjump`` section.
    """
    raw = load_yaml_mapping(Path(path))
    if "cpjump" not in raw:
        raise ValueError(f"Dataset config {path} has no 'cpjump' section")
    return CPJumpConfig.model_validate(raw["cpjump"])
