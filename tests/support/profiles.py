"""Builder for a synthetic on-disk profile tree the benchmark harness can read.

Real CPJUMP1 profiles are ~3 GB of 541-column CSVs. These are the same shape at
1/1000 the size: enough plates, replicates and negative controls for copairs to
form positive and negative pairs, with replicates of a perturbation deliberately
correlated so mAP clears the significance threshold.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.profiles import numbered_feature_columns, profile_path, write_profile

BATCH = "2020_11_04_CPJUMP1"
FEATURE_WIDTH = 24


@dataclass(frozen=True)
class Perturbation:
    """One perturbation and the gene it targets."""

    broad_sample: str
    gene: str


def _well(index: int) -> str:
    """Sequential well labels: A01, A02, ... B01, ..."""
    return f"{chr(ord('A') + index // 12):s}{index % 12 + 1:02d}"


def _signature(broad_sample: str, width: int) -> np.ndarray:
    """The perturbation's true profile, identical on every plate.

    Seeded from the sample name rather than a shared stream, because a
    perturbation that looked different on each plate would have no cross-plate
    replicate signal, which is exactly what replicability measures.
    """
    seed = int.from_bytes(sha256(broad_sample.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).normal(size=width)


def make_plate_profiles(
    plate: str,
    perturbations: Sequence[Perturbation],
    *,
    replicates: int = 3,
    negcon_wells: int = 4,
    width: int = FEATURE_WIDTH,
    noise: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """One plate of profiles where replicates of a perturbation cluster tightly.

    Each perturbation has a fixed signature shared across plates; its replicate
    wells are that signature plus *noise*. Negative controls are pure noise
    around zero, so they do not cluster with anything.

    Args:
        plate: Plate barcode, written to ``Metadata_Plate``.
        perturbations: Perturbations to lay out, ``replicates`` wells each.
        replicates: Wells per perturbation.
        negcon_wells: Extra ``negcon`` wells appended after the perturbations.
        width: Number of feature columns.
        noise: Standard deviation added to each replicate's signature. Small
            relative to the signature, so same-sample wells are the nearest
            neighbours of each other.
        seed: Seeds the per-well noise only.

    Returns:
        A frame of ``Metadata_*`` columns plus ``feature_0000..``.
    """
    rng = np.random.default_rng(seed)
    features = numbered_feature_columns(width)
    signatures = {p.broad_sample: _signature(p.broad_sample, width) for p in perturbations}

    rows: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    index = 0
    for pert in perturbations:
        for _ in range(replicates):
            rows.append(
                {
                    "Metadata_Plate": plate,
                    "Metadata_Well": _well(index),
                    "Metadata_broad_sample": pert.broad_sample,
                    "Metadata_gene": pert.gene,
                    "Metadata_control_type": np.nan,
                }
            )
            values.append(signatures[pert.broad_sample] + rng.normal(scale=noise, size=width))
            index += 1

    for _ in range(negcon_wells):
        rows.append(
            {
                "Metadata_Plate": plate,
                "Metadata_Well": _well(index),
                "Metadata_broad_sample": "DMSO",
                "Metadata_gene": np.nan,
                "Metadata_control_type": "negcon",
            }
        )
        values.append(rng.normal(scale=noise, size=width))
        index += 1

    return pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(np.vstack(values), columns=features)],
        axis=1,
    )


def write_profile_tree(
    root: Path,
    plates: Mapping[str, Sequence[Perturbation]],
    *,
    batch: str = BATCH,
    replicates: int = 3,
    negcon_wells: int = 4,
    seed: int = 0,
) -> Path:
    """Write ``<root>/<batch>/<plate>/<plate>_<suffix>.csv.gz`` for every plate.

    Each plate gets its own seed offset, so two plates carrying the same
    perturbation are not identical copies.

    Returns:
        ``root``.
    """
    for offset, (plate, perturbations) in enumerate(plates.items()):
        frame = make_plate_profiles(
            plate,
            perturbations,
            replicates=replicates,
            negcon_wells=negcon_wells,
            seed=seed + offset,
        )
        write_profile(frame, profile_path(root, batch, plate))
    return root


def make_experiments_frame(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """The CPJUMP1 experiment table columns the benchmark steps query."""
    return pd.DataFrame(
        rows,
        columns=["Assay_Plate_Barcode", "Perturbation", "Cell_type", "Time"],
    )
