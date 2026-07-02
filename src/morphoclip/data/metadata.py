"""CPJUMP1 metadata index: unified (plate, well) -> PerturbationInfo lookup.

Parses plate maps, barcode mappings, and external metadata TSVs,
then caches the full lookup for fast access during training.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import yaml

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType

logger = logging.getLogger(__name__)


def _read_tsv(path: Path) -> list[dict[str, str]]:

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _classify_pert_type(
    row: dict[str, str],
    compound_ids: set[str],
    crispr_ids: set[str],
    orf_ids: set[str],
) -> PerturbationType:
    """Determine perturbation type from a platemap row."""
    broad_sample = row.get("broad_sample", "")
    pert_type_str = row.get("pert_type", "").lower()
    control_type = row.get("control_type", "").lower()

    if control_type and "negcon" in control_type:
        return PerturbationType.NEGCON
    if control_type and "poscon" in control_type:
        return PerturbationType.POSCON
    if pert_type_str == "control" or not broad_sample:
        return PerturbationType.NEGCON

    if broad_sample in compound_ids:
        return PerturbationType.COMPOUND
    if broad_sample in crispr_ids:
        return PerturbationType.CRISPR
    if broad_sample in orf_ids:
        return PerturbationType.ORF

    if broad_sample.startswith("BRD-"):
        return PerturbationType.COMPOUND
    return PerturbationType.UNKNOWN


class MetadataIndex:
    """Unified metadata index for CPJUMP1.

    Merges plate maps, barcode mappings, and external metadata into a
    single lookup: ``(plate_barcode, well) -> PerturbationInfo``.

    Usage::

        index = MetadataIndex.from_config(Path("configs/dataset.yml"))
        info = index.lookup("BR00116991", "A01")
    """

    def __init__(
        self,
        barcode_to_platemap: dict[str, str],
        platemap_wells: dict[str, dict[str, dict[str, str]]],
        compound_meta: dict[str, dict[str, str]],
        crispr_meta: dict[str, dict[str, str]],
        orf_meta: dict[str, dict[str, str]],
    ) -> None:
        self._barcode_to_platemap = barcode_to_platemap
        self._platemap_wells = platemap_wells
        self._compound_meta = compound_meta
        self._crispr_meta = crispr_meta
        self._orf_meta = orf_meta

        self._compound_ids = set(compound_meta.keys())
        self._crispr_ids = set(crispr_meta.keys())
        self._orf_ids = set(orf_meta.keys())

        self._cache: dict[tuple[str, str], PerturbationInfo] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for barcode, platemap_name in self._barcode_to_platemap.items():
            wells = self._platemap_wells.get(platemap_name, {})
            for well, row in wells.items():
                self._cache[(barcode, well)] = self._build_info(row)

    def _build_info(self, row: dict[str, str]) -> PerturbationInfo:
        broad_sample = row.get("broad_sample", "")
        pert_type = _classify_pert_type(row, self._compound_ids, self._crispr_ids, self._orf_ids)
        control_type = row.get("control_type", "")

        kwargs: dict[str, str | PerturbationType] = {
            "pert_type": pert_type,
            "broad_sample": broad_sample,
            "control_type": control_type,
        }

        if pert_type == PerturbationType.COMPOUND and broad_sample in self._compound_meta:
            meta = self._compound_meta[broad_sample]
            kwargs["pert_iname"] = meta.get("pert_iname", "")
            kwargs["target_list"] = meta.get(
                "target", meta.get("target_list", meta.get("gene", ""))
            )
            kwargs["smiles"] = meta.get("smiles", "")
            kwargs["pubchem_cid"] = meta.get("pubchem_cid", "")
            kwargs["moa"] = meta.get("moa", "")

        elif pert_type == PerturbationType.CRISPR and broad_sample in self._crispr_meta:
            meta = self._crispr_meta[broad_sample]
            kwargs["gene"] = meta.get("gene", "")
            kwargs["pert_iname"] = meta.get("pert_iname", meta.get("gene", ""))
            kwargs["target_sequence"] = meta.get("target_sequence", "")
            kwargs["negcon_control_type"] = meta.get("negcon_control_type", "")

        elif pert_type == PerturbationType.ORF and broad_sample in self._orf_meta:
            meta = self._orf_meta[broad_sample]
            kwargs["gene"] = meta.get("gene", "")
            kwargs["pert_iname"] = meta.get("pert_iname", meta.get("gene", ""))
            kwargs["negcon_control_type"] = meta.get("negcon_control_type", "")

        return PerturbationInfo(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_config(cls, config_path: Path) -> MetadataIndex:
        """Build index from ``configs/dataset.yml``."""
        with open(config_path) as f:
            config = yaml.safe_load(f)["cpjump"]
        return cls.from_directory(Path(config["local"]["metadata"]), config["batch"])

    @classmethod
    def from_directory(cls, metadata_dir: Path, batch: str | None = None) -> MetadataIndex:
        """Build index from a metadata directory tree."""
        platemap_dir = metadata_dir / "platemaps"
        ext_dir = metadata_dir / "external_metadata"

        if batch is None:
            batch_dirs = [d for d in platemap_dir.iterdir() if d.is_dir()]
            if not batch_dirs:
                raise FileNotFoundError(f"No batch directories in {platemap_dir}")
            batch = batch_dirs[0].name
            logger.info("Auto-detected batch: %s", batch)

        batch_dir = platemap_dir / batch

        barcode_to_platemap = cls._load_barcode_map(batch_dir)
        platemap_wells = cls._load_platemaps(batch_dir)

        compound_targets = ext_dir / "JUMP-Target-1_compound_metadata_targets.tsv"
        compound_file = (
            "JUMP-Target-1_compound_metadata_targets.tsv"
            if compound_targets.exists()
            else "JUMP-Target-1_compound_metadata.tsv"
        )
        compound_meta = cls._load_external(ext_dir, compound_file)
        crispr_meta = cls._load_external(ext_dir, "JUMP-Target-1_crispr_metadata.tsv")
        orf_meta = cls._load_external(ext_dir, "JUMP-Target-1_orf_metadata.tsv")

        return cls(
            barcode_to_platemap=barcode_to_platemap,
            platemap_wells=platemap_wells,
            compound_meta=compound_meta,
            crispr_meta=crispr_meta,
            orf_meta=orf_meta,
        )

    @staticmethod
    def _load_barcode_map(batch_dir: Path) -> dict[str, str]:
        barcode_csv = batch_dir / "barcode_platemap.csv"

        if not barcode_csv.exists():
            logger.warning("barcode_platemap.csv not found at %s", barcode_csv)
            return {}

        result: dict[str, str] = {}

        for row in _read_csv(barcode_csv):
            barcode = row.get("Assay_Plate_Barcode", "")
            platemap_name = row.get("Plate_Map_Name", "")
            if barcode and platemap_name:
                result[barcode] = platemap_name

        logger.info("Loaded %d barcode mappings", len(result))
        return result

    @staticmethod
    def _load_platemaps(batch_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
        txt_files = sorted(batch_dir.glob("*.txt"))
        platemap_subdir = batch_dir / "platemap"
        if platemap_subdir.is_dir():
            txt_files = sorted(set(txt_files) | set(platemap_subdir.glob("*.txt")))

        result: dict[str, dict[str, dict[str, str]]] = {}
        for txt_file in txt_files:
            wells: dict[str, dict[str, str]] = {}

            for row in _read_tsv(txt_file):
                well = row.get("well_position", "")
                if well:
                    wells[well] = dict(row)

            result[txt_file.stem] = wells
            logger.debug("Loaded platemap %s with %d wells", txt_file.stem, len(wells))

        logger.info("Loaded %d platemap files", len(result))
        return result

    @staticmethod
    def _load_external(ext_dir: Path, filename: str) -> dict[str, dict[str, str]]:
        path = ext_dir / filename
        if not path.exists():
            logger.warning("External metadata not found: %s", path)
            return {}
        result: dict[str, dict[str, str]] = {}
        for row in _read_tsv(path):
            key = row.get("broad_sample", "")
            if key:
                result[key] = dict(row)
        logger.info("Loaded %d entries from %s", len(result), filename)
        return result

    def lookup(self, plate_barcode: str, well: str) -> PerturbationInfo:
        """Look up perturbation info for a (plate, well) pair."""
        well = well.upper()
        info = self._cache.get((plate_barcode, well))
        if info is not None:
            return info
        logger.debug("No metadata for plate=%s well=%s", plate_barcode, well)
        return PerturbationInfo()

    def wells_for_plate(self, plate_barcode: str) -> list[str]:
        """List all well positions for a given plate."""
        platemap_name = self._barcode_to_platemap.get(plate_barcode, "")
        return sorted(self._platemap_wells.get(platemap_name, {}).keys())

    def plates(self) -> list[str]:
        """List all plate barcodes in the index."""
        return sorted(self._barcode_to_platemap.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"MetadataIndex(plates={len(self._barcode_to_platemap)}, "
            f"wells={len(self._cache)}, "
            f"compounds={len(self._compound_meta)}, "
            f"crispr={len(self._crispr_meta)}, "
            f"orf={len(self._orf_meta)})"
        )
